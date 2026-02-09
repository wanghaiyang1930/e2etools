"""
@brief: Image classification dataset.
@author: wanghaiyang
@date: 2026-02-06
"""

from ntpath import isdir
import os

import torch
from torch.utils.data import Dataset

from PIL import Image

class ImageClassificationDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        print(f"Root directory: {self.root}")
        self.classes = sorted(os.listdir(self.root))
        self.class_to_idx = {name : i for i, name in enumerate(self.classes)}
        self.idx_to_class = {i : name for i, name in enumerate(self.classes)}

        self.images = []
        self.labels = []

        for dir_name in self.classes:
            sub_dir = os.path.join(self.root, dir_name)
            if os.path.isdir(sub_dir):
                for image_name in os.listdir(sub_dir):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(sub_dir, image_name))
                        self.labels.append(self.class_to_idx[dir_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.images):
            raise IndexError(f"Index out of images range, index: {index}, length: {len(self.images)}")

        if index >= len(self.labels):
            raise IndexError(f"Index out of labels range, index: {index}, length: {len(self.labels)}")

        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[index]

        data = {
            "index" : index,
            "image" : image,
            "label" : label,
            "lable_name" : self.idx_to_class[label],
            "image_name" : image_path
        }

        return data

if __name__ == "__main__":
    root_dir = "/home/wanghaiyang"

    dataset = ImageClassificationDataset(root_dir)

    print(f"Dataset length: {len(dataset)}")
    print(f"First item: {dataset[0]}")
    print(f"Second item: {dataset[1]}")
    print(f"Third item: {dataset[2]}")
    print("--------------------------------")
    print(f"Classses: {dataset.classes}")
    print(f"Class to index: {dataset.class_to_idx}")
    print(f"Index to class: {dataset.idx_to_class}")