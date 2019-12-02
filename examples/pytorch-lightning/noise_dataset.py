import torch.utils.data
import numpy as np
from sklearn.model_selection import train_test_split

N = 1000

X = np.concatenate((
    np.random.normal(0, 1, (N, 16)),
    np.random.normal(3, 2, (N, 16)),
    np.random.normal(5, 10, (N, 16)),
), axis=0).astype(np.float32)

Y = np.concatenate((
    np.repeat(0, N),
    np.repeat(1, N),
    np.repeat(2, N),
), axis=0)


train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.33)

train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.1)


class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, type="train"):
        if type == "train":
            self.X = train_X
            self.Y = train_Y
        elif type == "test":
            self.X = test_X
            self.Y = test_Y
        elif type == "validate":
            self.X = val_X
            self.Y = val_Y
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        assert len(self.X) == len(self.Y)
        return len(self.X)
