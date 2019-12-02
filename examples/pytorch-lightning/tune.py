from ray import tune
from argparse import ArgumentParser
from pytorch_lightning import Trainer

from noise import System

def to_argparse(config):
    params = ArgumentParser(add_help=False)
    for key in config:
        setattr(params, key, config[key])
    return params

def train(config):
    params = to_argparse(config)
    model = System(params)
    trainer = Trainer()
    trainer.fit(model)
    trainer.test(model)
    tune.track.log(accuracy=model.accuracy)

analysis = tune.run(
    train,
    config={
        "lr": 0.001,
        "batch_size": 32,
        "hidden": tune.grid_search([2, 4, 6, 8, 10])
    },   
)

print("Best config: ", analysis.get_best_config(metric="accuracy"))
df = analysis.dataframe()

print(df)