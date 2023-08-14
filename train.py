from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import Accuracy
import torch.nn as nn
import torch
import wandb

from resnet import get_pretrained_model
from datamodules import SamDataModule
from loggers import OutputLogger

pl.seed_everything(42)


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
        combined_tensor, _ = batch
        output = self.model(combined_tensor)

        return torch.argmax(output, dim=1)

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


if __name__ == "__main__":
    model = get_pretrained_model(n_classes=2)
    lightningmodule = LightningModule(model)

    datamodule = SamDataModule()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=[
            WandbLogger(project="Out Painting"),
        ],
        callbacks=[
            ModelCheckpoint(
                dirpath=wandb.run.dir,
                save_last=True,
            ),
            OutputLogger(),
        ],
        max_epochs=100,
        check_val_every_n_epoch=1,
        log_every_n_steps=1
    )

    trainer.fit(
        model=lightningmodule,
        datamodule=datamodule,
    )
