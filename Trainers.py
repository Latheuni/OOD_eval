## packages
import os
import wandb
import torch
from torch import nn
import pytorch_lightning as L
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import BasePredictionWriter


# Start weigts and biases
# wandb_logger = WandbLogger()
# trainer = Trainer(logger=wandb_logger)


class LitBasicNN(L.LightningModule):
    def __init__(self, NN, loss_function, learning_rate):
        super().__init__()
        self.NN = NN
        self.loss = loss_function
        self.lr = learning_rate

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def training_step(self, batch, batch_idx):
        x, y = batch
        scores = self.NN(x)
        loss = self.loss_function(scores, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        scores = self.NN(x)
        val_loss = self.loss_function(scores, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        scores = self.NN(x)
        test_loss = self.loss_function(scores, y)
        self.log("test_loss", test_loss)
        return scores, y

    def configure_optimizers(self):
        optimizer = torch.optimize.Adam(self.parameters, lr=self.lr)

    def return_net(self):
        return self.NN
