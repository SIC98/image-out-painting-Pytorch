from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import gradio as gr
from PIL import Image
import torch

from lightningmodule import LightningModule
from resnet import get_pretrained_model
from data import CustomSamDataset

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"


def predict(input_img, masks):
    trainer = pl.Trainer(accelerator="gpu")
    dataset = CustomSamDataset(input_img, masks)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = get_pretrained_model(n_classes=2)
    lightningmodule = LightningModule(model)

    y_hat = trainer.predict(
        lightningmodule,
        dataloader,
        return_predictions=True,
        ckpt_path="./wandb/run-20230814_210329-kt28nagp/files/epoch=99-step=3700.ckpt",
    )

    combined_y_hat = torch.cat(y_hat)
    indices_of_ones = combined_y_hat.nonzero(as_tuple=True)[0].tolist()

    return indices_of_ones


def mask_image(img, masks, indices_of_ones):

    masks = [masks[i] for i in indices_of_ones]
    mask = np.logical_or.reduce(masks)

    img_array = np.array(img)
    white_img = np.ones_like(img_array) * 255
    result_array = np.where(mask[..., None], img_array, white_img)
    result_img = Image.fromarray(np.uint8(result_array))

    return result_img


def sepia(input_img):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(input_img)
    masks = [mask['segmentation'] for mask in masks]

    output_img = Image.fromarray(input_img, "RGB")
    indices_of_ones = predict(output_img, masks)

    output_img = mask_image(output_img, masks, indices_of_ones)

    return output_img, output_img
