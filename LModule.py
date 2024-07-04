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
