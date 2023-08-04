from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageOps
import gradio as gr
import torch
import os

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
).to('cuda')


def image_mod(image, left_border, right_border, top_border, bottom_border):
    mask_image = Image.new('RGB', image.size, (0, 0, 0))

    left_border, right_border, top_border, bottom_border = int(left_border), int(right_border), int(top_border), int(bottom_border)

    w, h = image.size
    result_w = w + left_border + right_border
    result_h = h + top_border + bottom_border
    add_w = -result_w % 8
    add_h = -result_h % 8

    border = (left_border, top_border, right_border + add_w, bottom_border + add_h)

    image = ImageOps.expand(image, border, fill='white')
    mask_image = ImageOps.expand(mask_image, border, fill='white')

    width, height = image.size

    image = pipe(prompt='', image=image, mask_image=mask_image, height=height, width=width).images[0]

    return image.crop((0, 0, width - add_w, height - add_h))


demo = gr.Interface(
    fn=image_mod,
    inputs=[gr.Image(type="pil"), 'text', 'text', 'text', 'text'],
    outputs="image",
    # flagging_options=["blurry", "incorrect", "other"],
)

if __name__ == "__main__":
    demo.launch()