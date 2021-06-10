import os
import pandas as pd
import numpy as np
import pickle
import json

para = {
    "window_size": 1,
    "step_size": 0.5,
    "structured_file": "BGL.log_structured.csv",
    "BGL_sequence_train": "BGL_sequence_train.csv",
    "BGL_sequence_test": "BGL_sequence_test.csv"
}


def load_BGL():

    structured_file = para["structured_file"]
    # load data
    bgl_structured = pd.read_csv(structured_file)
    # get the label for each log("-" is normal, else are abnormal label)
    bgl_structured['Label'] = (bgl_structured['Label'] != '-').astype(int)
    return bgl_structured


def bgl_sampling(bgl_structured, phase="train"):

    label_data, time_data, event_mapping_data = bgl_structured['Label'].values, bgl_structured[
        'Timestamp'].values, bgl_structured['EventId'].values
    log_size = len(label_data)
    # split into sliding window
    start_time = time_data[0]
    start_index = 0
    end_index = 0
    start_end_index_list = []
    # get the first start, end index, end time
    for cur_time in time_data:
        if cur_time < start_time + para["window_size"]*3600:
            end_index += 1
            end_time = cur_time
        else:
            start_end_pair = tuple((start_index, end_index))
            start_end_index_list.append(start_end_pair)
            break
    while end_index < log_size:
        start_time = start_time + para["step_size"]*3600
        end_time = end_time + para["step_size"]*3600
        for i in range(start_index, end_index):
            if time_data[i] < start_time:
                i += 1
            else:
                break
        for j in range(end_index, log_size):
            if time_data[j] < end_time:
                j += 1
            else:
                break
        start_index = i
        end_index = j
        start_end_pair = tuple((start_index, end_index))
        start_end_index_list.append(start_end_pair)
    # start_end_index_list is the  window divided by window_size and step_size,
    # the front is the sequence number of the beginning of the window,
    # and the end is the sequence number of the end of the window
    inst_number = len(start_end_index_list)
    print('there are %d instances (sliding windows) in this dataset' % inst_number)

    # get all the log indexs in each time window by ranging from start_index to end_index

    expanded_indexes_list = [[] for i in range(inst_number)]
    expanded_event_list = [[] for i in range(inst_number)]

    for i in range(inst_number):
        start_index = start_end_index_list[i][0]
        end_index = start_end_index_list[i][1]
        if start_index > end_index:
            continue
        for l in range(start_index, end_index):
            expanded_indexes_list[i].append(l)
            expanded_event_list[i].append(event_mapping_data[l])
    #=============get labels and event count of each sliding window =========#

    labels = []

    for j in range(inst_number):
        label = 0  # 0 represent success, 1 represent failure
        for k in expanded_indexes_list[j]:
            # If one of the sequences is abnormal (1), the sequence is marked as abnormal
            if label_data[k]:
                label = 1
                continue
        labels.append(label)
    assert inst_number == len(labels)
    print("Among all instances, %d are anomalies" % sum(labels))

    BGL_sequence = pd.DataFrame(columns=['sequence', 'label'])
    BGL_sequence['sequence'] = expanded_event_list
    BGL_sequence['label'] = labels
    BGL_sequence.to_csv(para["BGL_sequence_{0}".format(phase)], index=False)
    return BGL_sequence


if __name__ == "__main__":
    bgl_structured = load_BGL()
    n_logs = len(bgl_structured)

    events = bgl_structured["EventTemplate"].values
    event_ids = bgl_structured["EventId"].values
    events_df = pd.read_csv("../bgl/templates.csv", memory_map=True)
    events_df = events_df.to_dict('records')
    event_map = {}
    for e in events_df:
        event_map[e['template']] = str(e['id'])

    ids_map = {}
    max_ids = 0
    # for i in range(len(events)):
    #     ids_map[event_ids[i]] = event_map[events[i]]

    n_session = 0
    train_logs = bgl_sampling(bgl_structured[:n_logs * 80 // 100], "train")
    train_logs = train_logs.to_dict("records")
    n_session += len(train_logs)
    list_events = []
    with open("bgl_train", mode="w") as f:
        for log in train_logs:
            # only train normal logs
            log_seq = log['sequence']
            if len(log_seq) == 0:
                continue
            # seq = []
            for x in log_seq:
                if x not in ids_map.keys():
                    max_ids += 1
                    ids_map[x] = max_ids
            seq = [str(ids_map[x]) for x in log_seq]
            list_events.extend(seq)
            if log['label'] == 1:
                continue
            seq = " ".join(seq)
            f.write(seq + "\n")

    print(max_ids)

    test_logs = bgl_sampling(bgl_structured[n_logs * 80 // 100:], "test")
    test_logs = test_logs.to_dict("records")
    n_session += len(test_logs)
    normal_test, abnormal_test = [], []
    for log in test_logs:
        # only train normal logs

        log_seq = log['sequence']
        if len(log_seq) == 0:
            continue

        for x in log_seq:
            if x not in ids_map.keys():
                max_ids += 1
                ids_map[x] = max_ids
        seq = [str(ids_map[x]) for x in log_seq]
        seq = " ".join(seq)

        if log['label'] == 1:
            abnormal_test.append(seq)
        else:
            normal_test.append(seq)
    print(n_session)
    with open("bgl_test_normal", mode="w") as f:
        [f.write(x + "\n") for x in normal_test]

    with open("bgl_test_abnormal", mode="w") as f:
        [f.write(x + "\n") for x in abnormal_test]

    # with open("../bgl/bgl_embeddings.json", "r") as f:
    #     embs = json.load(f)
    # print(embs.keys())
    # train_events = {}
    # list_events = list(set(list_events))
    # # print(list_events)
    # # print(len(list_events))
    # for k, v in embs.items():
    #     # print(k in list_events)
    #     if k in list_events:
    #         train_events[k] = v
    # print(train_events)
    # with open("train_embeddings.json", "w") as write_file:
    #     json.dump(train_events, write_file, indent=4)