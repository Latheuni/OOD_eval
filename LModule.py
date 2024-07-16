import torch
import torchmetrics
if torchmetrics.__version__ > "0.9.3":
    from torchmetrics.classification import (
        MulticlassAccuracy,
    )
else:
    from torchmetrics.classification import (
        Accuracy,
    )

import pytorch_lightning as L
from torchmetrics.classification import (
    Accuracy,
) 
import torch.nn.functional as F

import numpy as np
from torch import nn
from torch import autograd
from torch.distributions.dirichlet import Dirichlet
from Losses import CE_loss, UCE_loss
#from Normalizing_flows import *
class LitBasicNN(L.LightningModule):
    def __init__(self, NN, loss_function, learning_rate, n_classes, decay=0.95):
        super().__init__()
        self.NN = NN
        self.loss_function = loss_function
        self.lr = learning_rate
        self.decay = decay

        if torchmetrics.__version__ > "0.9.3":
            self.accuracy = MulticlassAccuracy(
                num_classes=n_classes, average="micro"
            )  # is not callable
            self.balanced_accuracy = MulticlassAccuracy(
                num_classes=n_classes, average="macro"
            )
            self.train_accuracy = MulticlassAccuracy(
                num_classes=n_classes, average="micro"
            )
            self.val_accuracy = MulticlassAccuracy(
                num_classes=n_classes, average="micro"
            )
            self.test_accuracy = MulticlassAccuracy(
                num_classes=n_classes, average="micro"
            )
            self.val_balanced_accuracy = MulticlassAccuracy(
                num_classes=n_classes, average="macro"
            )
            self.test_balanced_accuracy = MulticlassAccuracy(
                num_classes=n_classes, average="macro"
            )
        else:
            self.train_accuracy = Accuracy(num_classes=n_classes, average="micro")
            self.val_accuracy = Accuracy(num_classes=n_classes, average="micro")
            self.test_accuracy = Accuracy(num_classes=n_classes, average="micro")
            self.val_balanced_accuracy = Accuracy(
                num_classes=n_classes, average="macro"
            )
            self.test_balanced_accuracy = Accuracy(
                num_classes=n_classes, average="macro"
            )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        scores = self.NN(x)
        loss = self.loss_function(scores, y)
        scores = F.softmax(scores, dim=-1)
        self.log(
            "train_loss", loss, on_step=True
        )  # on_epoch acculumate and rduces all metric to the end of the epoch, on_step that specific call will not accumulate metrics
        self.train_accuracy(scores, y)
        self.log("training accuracy", self.train_accuracy, on_step=True)
        return loss # necessary o, ùpdule

    def validation_step(self, batch, batch_idx):
        x, y = batch
        scores = self.NN(x)
        val_loss = self.loss_function(scores, y)
        scores = F.softmax(scores, dim=-1)
        self.log("val_loss", val_loss, on_step=True)
        self.val_accuracy(scores, y)
        self.val_balanced_accuracy(scores, y)
        self.log("validation accuracy", self.val_accuracy, on_step=True)
        self.log(
            "validation balanced accuracy", self.val_balanced_accuracy, on_step=True
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        scores = self.NN(x)
        test_loss = self.loss_function(scores, y)
        scores = F.softmax(scores, dim=-1)
        self.log("test_loss", test_loss, on_step=True)
        self.test_accuracy(scores, y)
        self.test_balanced_accuracy(scores, y)
        self.log("test accuracy", self.test_accuracy, on_step=True)
        self.log("test balanced accuracy", self.test_balanced_accuracy, on_step=True)
        if batch_idx == 0:
            self.ytrue = y
            self.scores = scores 
        else:
            self.ytrue = torch.cat((self.ytrue, y), 0)
            self.scores = torch.cat((self.scores, scores), 0)
        return scores, y

    def predict_step(
        self, batch, batch_idx
    ):  # Loss needs to be minimized, max scores are correct label
        x, y = batch
        scores = self.NN(x)
        scores = F.softmax(scores, dim=-1)
        if batch_idx == 0:
            self.predictions = torch.argmax(scores, dim=1)
        else:
            self.predictions = torch.cat(
                (self.predictions, torch.argmax(scores, dim=1)), 0
            )
        return torch.argmax(scores, dim=1)

    def configure_optimizers(self):
        lambd = lambda epoch: self.decay
        
        optimizer = torch.optim.Adam(
            self.parameters(), lr=float(self.lr)
        )  # can still add weight decay

        # Don't use a lr_scheduler, does not improve results
        #lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        #    optimizer, lr_lambda=lambd
        #)
        return optimizer

## The following code is based on code from:  https://github.com/sharpenb/Posterior-Network/tree/main
__budget_functions__ = {'one': lambda N: torch.ones_like(N),
                        'log': lambda N: torch.log(N + 1.),
                        'id': lambda N: N,
                        'id_normalized': lambda N: N / N.sum(),
                        'exp': lambda N: torch.exp(N),
                        'parametrized': lambda N: torch.nn.Parameter(torch.ones_like(N).to(N.device))}

class LitPostNN(L.LightningModule):
    def __init__(self,N, # Count of data from each class in training set. list of ints
                NN, # network
                output_dim,
                batch_size = 128,  # Batch size. int
                budget_function='id',  # Budget function name applied on class count. name
                lr=1e-4,  # Learning rate. float
                loss='UCE',  # Loss name. string
                regr=1e-5,  # Regularization factor in Bayesian loss. float
                seed=123, 
                decay=0.95):  # Random seed for init. int
        super().__init__()
        if budget_function in __budget_functions__:
            self.N, self.budget_function = __budget_functions__[budget_function](N), budget_function
        else:
            raise NotImplementedError
        self.batch_size, self.lr = batch_size, lr
        self.loss_name = loss
        if self.loss_name == "UCE":
            self.loss = UCE_loss(output_dim)
        else:
            self.loss = CE_loss()
        self.regr = regr
        self.decay = decay
        self.save_hyperparameters()
        self.NN = NN

    def training_step(self, batch, batch_idx):
        x,y = batch
        if self.loss_name == 'CE':
            scores = self.NN(x)
            l = self.loss(scores,x)
            return l
        elif self.loss_name == "UCE":
            alpha, scores = self.NN(x)
            l = self.loss(alpha,x)
            return l
        else:
            raise NotImplementedError
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.loss_name == 'CE':
            scores = self.NN(x)
            val_loss = self.loss(scores,x)
        elif self.loss_name == "UCE":
            alpha, scores = self.NN(x)
            val_loss = self.loss(alpha,x)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        scores = self.NN(x)
        if self.loss_name == 'CE':
            scores = self.NN(x)
            test_loss = self.loss(scores,x)
        elif self.loss_name == "UCE":
            alpha, scores = self.NN(x)
            test_loss = self.loss(alpha,x)
        self.log("test_loss", test_loss)
        return scores, y

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        lambd = lambda epoch: self.decay
        
        optimizer = torch.optim.Adam(
            self.parameters(), lr=float(self.lr) #TODO: which part has trainable parameters?
        )  # can still add weight decay

        # Don't use a lr_scheduler, does not improve results
        #lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        #    optimizer, lr_lambda=lambd
        #)
        return optimizer

    