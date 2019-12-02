import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from argparse import ArgumentParser

from pytorch_lightning import Trainer
import pytorch_lightning as pl

from classifier import Classifier


class System(Classifier):
    def __init__(self, hparams):
        super(System, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(784),
            torch.nn.Linear(784, hparams.hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hparams.hidden, hparams.hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hparams.hidden, 10),
            torch.nn.Softmax(dim=1)
        )


        self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(x.shape[0], -1)[0,:])
        ])

        self.hparams = hparams



    def forward(self, x):
        return self.net(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=self.transforms), batch_size=self.hparams.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=self.transforms), batch_size=self.hparams.batch_size)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=self.transforms), batch_size=self.hparams.batch_size)



if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--hidden", type=int, default=50)

    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=None)

    args = parser.parse_args()

    model = System(args)
    trainer = Trainer(gpus=args.gpus)    
    trainer.fit(model)
    trainer.test(model)
