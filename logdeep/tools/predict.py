#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
from collections import Counter
import json

sys.path.append('../../')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from logdeep.dataset.log import log_dataset
from logdeep.dataset.sample import session_window
from logdeep.tools.utils import (save_parameters, seed_everything,
                                 train_val_split)


class Predicter():
    def __init__(self, model, options):
        self.data_dir = options['data_dir']
        self.device = options['device']
        self.model = model
        self.model_path = options['model_path']
        self.window_size = options['window_size']
        self.num_candidates = options['num_candidates']
        self.num_classes = options['num_classes']
        self.input_size = options['input_size']
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.batch_size = options['batch_size']
        self.sequence_length = options['sequence_length']
        self.vectorize()

    def vectorize(self):
        with open(self.data_dir + "bgl/bgl_embeddings.json", "r") as f:
            self.all_events = json.load(f)

        with open(self.data_dir + "bgl/train_embeddings.json", "r") as f:
            self.trained_events = json.load(f)

    def matching(self, u):
        dist = 10000000
        key = -1
        for k, v in self.trained_events.items():
            d = np.linalg.norm(np.array(u) - np.array(v))
            if d < dist:
                dist = d
                key = k
        return key

    def generate(self, name):
        window_size = self.window_size
        hdfs = {}
        length = 0
        with open(self.data_dir + name, 'r') as f:
            for ln in f.readlines():
                ln = list(map(lambda n: n, map(int, ln.strip().split())))
                ln = [-1] * (window_size + 1 - len(ln)) + ln
                hdfs[tuple(ln)] = hdfs.get(tuple(ln), 0) + 1
                length += 1
        print('Number of sessions({}): {}'.format(name, len(hdfs)))
        return hdfs, length

    def predict_unsupervised(self, normal_fname='hdfs_test_normal', abnormal_fname='hdfs_test_abnormal'):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_normal_loader, test_normal_length = self.generate(normal_fname)
        test_abnormal_loader, test_abnormal_length = self.generate(abnormal_fname)
        TP = 0
        FP = 0
        # Test the model
        start_time = time.time()
        with torch.no_grad():
            for line in tqdm(test_normal_loader.keys()):
                for i in range(0, len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    if label == -1:
                        print(line)
                    s_label = str(label)
                    if s_label not in self.trained_events.keys():
                        label = int(self.matching(self.all_events[s_label]))
                    temp_seq = []
                    for l in seq0:
                        s_l = str(l)
                        if s_l in self.trained_events.keys() or l == -1:
                            temp_seq.append(l)
                        else:
                            temp_seq.append(int(self.matching(self.all_events[s_l])))

                    seq0 = temp_seq.copy()

                    seq1 = [0] * self.sequence_length
                    log_conuter = Counter(seq0)
                    for key in log_conuter:
                        seq1[key] = log_conuter[key]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(
                        -1, self.num_classes, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0, seq1], device=self.device)
                    predicted = torch.argsort(output,
                                              1)[0][-self.num_candidates:]
                    if label not in predicted:
                        FP += test_normal_loader[line]
                        break
        with torch.no_grad():
            for line in tqdm(test_abnormal_loader.keys()):
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    s_label = str(label)
                    if s_label not in self.trained_events.keys():
                        label = int(self.matching(self.all_events[s_label]))
                    temp_seq = []
                    for l in seq0:
                        s_l = str(l)
                        if s_l in self.trained_events.keys() or l == -1:
                            temp_seq.append(l)
                        else:
                            temp_seq.append(int(self.matching(self.all_events[s_l])))

                    seq0 = temp_seq.copy()
                    seq1 = [0] * self.sequence_length
                    log_conuter = Counter(seq0)
                    for key in log_conuter:
                        seq1[key] = log_conuter[key]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(
                        -1, self.num_classes, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0, seq1], device=self.device)
                    predicted = torch.argsort(output,
                                              1)[0][-self.num_candidates:]
                    if label not in predicted:
                        TP += test_abnormal_loader[line]
                        break

        # Compute precision, recall and F1-measure
        FN = test_abnormal_length - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(FP, FN, P, R, F1))
        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))

    def predict_supervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_logs, test_labels = session_window(self.data_dir, datatype='test')
        test_dataset = log_dataset(logs=test_logs,
                                   labels=test_labels,
                                   seq=self.sequentials,
                                   quan=self.quantitatives,
                                   sem=self.semantics)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      pin_memory=True)
        tbar = tqdm(self.test_loader, desc="\r")
        TP, FP, FN, TN = 0, 0, 0, 0
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().to(self.device))
            output = self.model(features=features, device=self.device)
            output = F.sigmoid(output)[:, 0].cpu().detach().numpy()
            # predicted = torch.argmax(output, dim=1).cpu().numpy()
            predicted = (output < 0.2).astype(int)
            label = np.array([y.cpu() for y in label])
            TP += ((predicted == 1) * (label == 1)).sum()
            FP += ((predicted == 1) * (label == 0)).sum()
            FN += ((predicted == 0) * (label == 1)).sum()
            TN += ((predicted == 0) * (label == 0)).sum()
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(FP, FN, P, R, F1))