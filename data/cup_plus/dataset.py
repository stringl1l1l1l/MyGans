from torch.utils.data import Dataset, DataLoader
import os
import torch

class Cup(Dataset):
    def __init__(self, path="./tensor_data/pos", transform=None, label=0):
        self.label = label
        self.data = []
        self.transform = transform
        for filename in os.listdir(path):  # 读取文件夹里的tensor图片
            if filename.endswith('.pt'):
                tensor_path = os.path.join(path, filename)
                tensor = torch.load(tensor_path)
                self.data.append(tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label  # 人为给定标签
        if self.transform:  # 数据预处理/增强
            x = self.transform(x)
        return x, y


