import torch
import numpy as np
from torch.utils.data import Dataset


class Task_Dataset(Dataset):
    def __init__(self, X, y, task_id):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()  # convert to torch.Tensor here
        self.task_id = task_id
        assert self.X.shape[0] == self.y.shape[0]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        t = self.task_id[idx]
        return X, y, t


class TaskBatched_Dataset(Dataset):
    def __init__(self, X, y, task_id, batch_size, shuffle):
        sort_idx = np.argsort(task_id)
        self.X = X[sort_idx]
        self.y = y[sort_idx]
        self.task_id = task_id[sort_idx]
        self.tasks = np.unique(task_id)
        self.task_idx = [np.where(self.task_id == task)[0] for task in self.tasks]

        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.y) // self.batch_size

    def __getitem__(self, idx):
        if self.shuffle:
            task_id = np.random.choice(self.tasks)  # choose one task at random
            idx = np.random.choice(self.task_idx[task_id], self.batch_size)
        else:
            idx = slice(idx * self.batch_size, (idx + 1) * self.batch_size)  # not perfect at boundaries between tasks!!
        X = self.X[idx]
        y = self.y[idx]
        t = self.task_id[idx]
        return X, y, t


def balanced_subsample(x, y, subsample_size=1.0):
    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems is None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems * subsample_size)

    xs = []
    ys = []

    for ci, this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.full(use_elems, ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs, ys


def zscore(X):
    X -= np.mean(X, axis=0, keepdims=True)
    X /= np.std(X, axis=0, keepdims=True)
    return X
