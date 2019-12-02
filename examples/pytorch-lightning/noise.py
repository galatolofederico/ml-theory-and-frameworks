import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl


from classifier import Classifier
from noise_dataset import NoiseDataset


class System(Classifier):

    def __init__(self, hparams):
        super(System, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(16),
            torch.nn.Linear(16, hparams.hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hparams.hidden, hparams.hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hparams.hidden, 3),
            torch.nn.Softmax(dim=1)
        )

        self.hparams = hparams



    def forward(self, x):
        return self.net(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(NoiseDataset(type="train"), batch_size=self.hparams.batch_size)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(NoiseDataset(type="test"), batch_size=self.hparams.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(NoiseDataset(type="validate"), batch_size=self.hparams.batch_size)


if __name__ == "__main__":

    parser = ArgumentParser(add_help=False)

    parser.add_argument("--hidden", type=int, default=10)

    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--batch-size", type=int, default=32)

    model = System(parser.parse_args())

    trainer = Trainer() 
    trainer.fit(model)
    trainer.test(model)
