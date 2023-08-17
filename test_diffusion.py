from diffusers import StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler
import torchvision.transforms as T
import pytorch_lightning as pl

import torch

from PIL import Image, ImageOps
import numpy as np

pl.seed_everything(42)

# pipe = StableDiffusionInpaintPipeline.from_single_file(
#     './civit_ai/dreamshaper_8Inpainting.safetensors',
#     torch_dtype=torch.float16,
# )

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    'stabilityai/stable-diffusion-2-inpainting',
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipe.to('cuda')

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config)
# Todo: Use DPM

# Linear -> Cosine

prompt = ''  # A picture

image = Image.open('./images/dog.png')

# # tensor = torch.linspace(255, 0, steps=256)
# tensor = torch.zeros(256, dtype=torch.float32)

# x = torch.tensor([255] * 256)

y = torch.cat((
    # torch.full((128-32,), 0, dtype=torch.float32),
    # torch.cos(torch.linspace(0, 1, steps=32,
    #           dtype=torch.float32) * np.pi) / -2 + 0.5,
    # torch.full((384,), 1, dtype=torch.float32),
    torch.full((128,), 0, dtype=torch.float32),
    torch.full((384,), 1, dtype=torch.float32),
), dim=0)
transform = T.ToPILImage()
y = y.repeat(3, 512, 1)

mask_image = transform(y)


# mask_image = Image.new('RGB', image.size, (10, 10, 10))
# print('original image size: ', image.size)

border = (0, 0, 384, 0)
image = ImageOps.expand(image, border, fill='black')
# mask_image = ImageOps.expand(mask_image, border, fill='white')
# print(mask_image.size)

image = image.crop((384, 0, 512 + 384, 512))
# mask_image = mask_image.crop((256, 0, 512 + 256, 512))


image.save('./cropped_image.png')
mask_image.save("./mask.png")


# mask_image_np = np.array(mask_image)

# print(mask_image_np.shape)

# mask_image_np[:, mask_image.width//2:] = (255, 255, 255)  # Make the right half of the mask white

# # Convert the mask back to PIL image
# mask_image = Image.fromarray(mask_image_np)


# image and mask_image should be PIL images.
# The mask structure is white for inpainting and black for keeping as is
result = pipe(prompt=prompt, image=image, mask_image=mask_image,
              height=512, width=512, num_inference_steps=50).images[0]

result = result.crop((128, 0, 512, 512))

image.paste(result, (128, 0))

image.save("./result.png")
