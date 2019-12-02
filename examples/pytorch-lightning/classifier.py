import torch
import pytorch_lightning as pl


def mean_lod(key, lod):
    ret = 0
    for d in lod:
        ret += d[key]
    return ret/len(lod)


class Classifier(pl.LightningModule):
    def __init__(self):
        super(Classifier, self).__init__()

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        
        acc = (out.max(dim=1).indices == y).float().mean()
        loss = self.loss_fn(out, y)

        return {
            "loss": loss,
            "progress_bar": {
                "loss": loss,
                "acc": acc,
            },
            "log": {
                "loss": loss.item(),
                "acc": acc.item()
            }
        }

    def validation_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)

        rights = (out.max(dim=1).indices == y).float().sum()
        loss = self.loss_fn(out, y)

        return {
            "size": out.shape[0],
            "loss": loss.item(),
            "rights": rights.item()
        }
    
    def validation_end(self, outputs):
        acc = 0
        tots = 0
        for output in outputs:
            acc += output["rights"]
            tots += output["size"]
        acc = acc/tots
        
        return{
            "log":{
                "val_accuracy": acc,
                "val_loss": mean_lod("loss", outputs),
            }
        }

    def test_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        
        rights = (out.max(dim=1).indices == y).float().sum()

        return {
            "size": out.shape[0],
            "rights": rights.item(),
            "accuracy": rights.item()/out.shape[0]
        }
    
    def test_end(self, outputs):
        acc = 0
        tots = 0
        for output in outputs:
            acc += output["rights"]
            tots += output["size"]
        print("Accuracy :", acc/tots)
        
        self.accuracy = acc/tots

        return {
            "log":{
                "test_accuracy": acc/tots
            },
            "progress_bar": {
                "test_accuracy": acc/tots
            }
        }

