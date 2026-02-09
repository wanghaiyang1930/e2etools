"""
"""

import torch
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, size=100):
        # [size, channel, width, height]
        self.datas = torch.randn(size, 3, 32, 32)
        self.labels = torch.randint(0, 10, (size,))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        item = {
            "data": self.datas[idx],
            "label": self.labels[idx]
        }
        return item

dataset = SimpleDataset(100)

print(f"Dataset length: {len(dataset)}")
print(f"Dataset first item: {dataset[0]["data"].shape}")

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    drop_last=False
)

for idx, items in enumerate(dataloader):
    print(f"Batch size: {idx}")
    print(f"Items type: {type(items)}")
    print(f"Items length: {len(items)}")

    print(f"Data shape: {items["data"].shape}")
    print(f"Label shape: {items["label"].shape}")