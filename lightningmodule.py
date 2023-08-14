from torchmetrics.classification import Accuracy
import pytorch_lightning as pl
import torch.nn as nn
import torch


class LightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module
    ):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=2)
        self.valid_accuracy = Accuracy(task="multiclass", num_classes=2)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.train_accuracy(y_hat, y)

        return {
            "loss": loss,
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.valid_accuracy(y_hat, y)

        return {
            "loss": loss,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        y_hat = self.model(x)

        return torch.argmax(y_hat, dim=1)

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_accuracy.compute())
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        self.log("valid_acc", self.valid_accuracy.compute())
        self.valid_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=1e-3,
        )

        return optimizer
