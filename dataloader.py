import numpy as np
import torch
from torch.utils.data import Dataset
import random


def get_sample_index(GroundTruth, SampleNumberPerClass):
    # Data: H * W * C
    class_number = int(GroundTruth.max())
    all_indices = []

    for N in range(1, class_number + 1):
        idx = np.argwhere(GroundTruth == N)
        if idx.shape[0] >= SampleNumberPerClass:
            all_indices.extend(idx[np.random.choice(idx.shape[0], SampleNumberPerClass, replace=False), :])
        else:
            temp = SampleNumberPerClass
            while temp > 0:
                all_indices.extend(
                    idx if temp > idx.shape[0] else idx[np.random.choice(idx.shape[0], temp, replace=False), :])
                temp -= idx.shape[0]

    random.shuffle(all_indices)
    return all_indices


class ListDataset(Dataset):
    def __init__(self, data, label, sample_idx, patch_size):
        self.data = data
        self.label = label
        self.patch_size = patch_size
        self.indices = sample_idx
        self.HalfWidth = self.patch_size // 2


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        centerX, centerY = self.indices[idx][0], self.indices[idx][1]
        data = np.transpose(self.data[centerX - self.HalfWidth: centerX + self.HalfWidth + 1, \
                                      centerY - self.HalfWidth: centerY + self.HalfWidth + 1, :],
                                      (2, 0, 1))
        label = self.label[centerX][centerY]
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        return data, label


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.data, self.label = next(self.loader)

        except StopIteration:
            self.next_input = None

            return
        with torch.cuda.stream(self.stream):
            self.data = self.data.cuda(non_blocking=True)
            self.label = self.label.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.data
        label = self.label

        self.preload()
        return data, label
