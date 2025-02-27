import os 
import torch
import pandas as pd
import numpy as np 
from torch.utils.data import Dataset

class DKiTDataset(Dataset):
    def __init__(self, split_file, printer=print):
        data = pd.read_csv(split_file)
        self.X, self.y, self.M = self.build_data(data)
        printer(f"Dataset loaded from {split_file}, X: {self.X.shape}, y: {self.y.shape}, M: {self.M.shape}")

    def build_data(self, data):
        X = data.iloc[:, 3:19].values  # (16,)
        y = data.iloc[:, 19:35].values  # (16,)
        M = (y >= 0).astype(X.dtype)  # (16,)
        return torch.tensor(X, dtype=torch.float32), \
                torch.tensor(y, dtype=torch.float32), \
                torch.tensor(M, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.M[idx]
    
class MLDataset(Dataset):
    def __init__(self, split_file, printer=print):
        data = pd.read_csv(split_file)
        self.X, self.M, self.y = self.build_data(data)
        printer(f"Dataset loaded from {split_file}, X: {self.X.shape}, y: {self.y.shape}")

    def build_data(self, data):
        X = data.iloc[:, 3:19].values  # (16,)
        y = data.iloc[:, 19:35].values  # (16,)
        M = (y >= 0).astype(X.dtype)  # (16,) only one position has the efficient value (0/1)
        ML_X = torch.concat([torch.tensor(X, dtype=torch.float32), torch.tensor(M, dtype=torch.float32)], dim=1) # (N, 32)
        ML_y = torch.LongTensor(np.max(y, axis=1)) # (N,)
        return ML_X, torch.tensor(M, dtype=torch.float32), ML_y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.M[idx]
    
if __name__ == '__main__':
    pass

