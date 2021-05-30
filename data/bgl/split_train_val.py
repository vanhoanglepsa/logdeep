from sklearn.utils import shuffle

with open("bgl_train", mode="r") as f:
    data = f.readlines()
    data = [x.strip() for x in data]

data = shuffle(data)
n_samples = len(data)
print(n_samples)
train = data[:n_samples * 90 // 100]
val = data[n_samples * 90 // 100:]
with open("bgl_train", mode="w") as f:
    [f.write(x + "\n") for x in train]

with open("bgl_val", mode="w") as f:
    [f.write(x + "\n") for x in val]