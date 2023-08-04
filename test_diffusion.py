from diffusers import StableDiffusionInpaintPipeline
import torch

from PIL import Image, ImageOps
import numpy as np

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    # "stabilityai/stable-diffusion-2-inpainting",
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
).to('cuda')

prompt = ""

image = Image.open('./image1.png')
mask_image = Image.new('RGB', image.size, (0, 0, 0))
print('original image size: ', image.size)

border = (56+192, 56-40, 56+192, 56)  # left, top, right, bottom
image = ImageOps.expand(image, border, fill='white')
image.save("./image1-new.png")
print(image.size)
# mask_image = Image.open('./image2.png')

# image = image.resize((512, 512))

# x = np.array(image)
# print(x.shape)
# y = np.array(mask_image)

# print(x.shape, y.shape)

# print(y)

mask_image = ImageOps.expand(mask_image, border, fill='white')
print(mask_image.size)
# mask_image_np = np.array(mask_image)

# print(mask_image_np.shape)

# mask_image_np[:, mask_image.width//2:] = (255, 255, 255)  # Make the right half of the mask white

# # Convert the mask back to PIL image
# mask_image = Image.fromarray(mask_image_np)


#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
image = pipe(prompt=prompt, image=image, mask_image=mask_image, height=584, width=1008).images[0]
image.save("./new_cube.png")

