"""
@brief: Simple dataset.
@author: wanghaiyang
@date: 2026-02-06
"""

import torch
from torch.utils.data import Dataset

class SampleDataset(Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):

        if index >= len(self.datas):
            raise IndexError("Index out of range")

        data = {
            "data": self.datas[index],
            "label": self.labels[index]
        }
        return data

if __name__ == "__main__":
    datas = [1, 2, 3, 4, 5]
    labels = [1, 2, 3, 4, 5]

    dataset = SampleDataset(datas, labels)
    
    print("dataset length: ", len(dataset))
    print("first item: ", dataset[0])