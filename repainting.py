from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from PIL import Image
import torch
import cv2

from lightningmodule import LightningModule
from resnet import get_pretrained_model
from data import CustomSamDataset
from out_painting import crop_image, merge_image

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


def paint(input_img, mask):
    mask = mask.astype(np.uint8) * 255
    mask = Image.fromarray(mask, "L")

    w, h = input_img.size
    w -= 512
    h -= 512

    for i in range(0, w+256, 256):
        if i > w:
            i = w
        for j in range(0, h+256, 256):
            if j > h:
                j = h

            generated_image = crop_image(
                input_img, mask, i, j, i+512, j+512)
            image, mask = merge_image(
                input_img, generated_image, mask, i, j)

    return image


def mask_image(img, masks, indices_of_ones):

    masks = [masks[i] for i in indices_of_ones]
    mask = np.logical_or.reduce(masks)
    binarized = (mask.astype(np.uint8) * 255)

    kernel = np.ones((20, 20), np.uint8)
    mask = cv2.dilate(binarized, kernel, iterations=1)
    mask = mask == 255

    img_array = np.array(img)
    white_img = np.ones_like(img_array) * 255
    result_array = np.where(mask[..., None], img_array, white_img)
    result_img = Image.fromarray(np.uint8(result_array))

    return result_img, mask


def repaint(input_img):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(input_img)
    segmentation = [mask['segmentation'] for mask in masks]

    input_img = Image.fromarray(input_img, "RGB")
    indices_of_ones = predict(input_img, segmentation)

    maksed_img, mask = mask_image(input_img, segmentation, indices_of_ones)

    repaint_image = paint(input_img, mask)

    return maksed_img, repaint_image
