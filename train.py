import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

from lightningmodule import LightningModule
from resnet import get_pretrained_model
from loggers import OutputLogger
from data import SamDataset


def main():
    pl.seed_everything(42)

    model = get_pretrained_model(n_classes=2)
    lightningmodule = LightningModule(model)

    train_dataset = SamDataset(type="train")
    valid_dataset = SamDataset(type="validation")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=16, shuffle=False
    )

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
        lightningmodule,
        train_dataloader,
        valid_dataloader
    )


if __name__ == "__main__":
    main()
