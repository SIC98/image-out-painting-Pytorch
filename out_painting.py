from diffusers import StableDiffusionInpaintPipeline
from diffusers import EulerAncestralDiscreteScheduler
from PIL import Image, ImageOps
import numpy as np
import torch

# pipe = StableDiffusionInpaintPipeline.from_single_file(
#     "./civit_ai/dreamshaper_8Inpainting.safetensors",
#     torch_dtype=torch.float16,
# ).to("cuda")

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
).to("cuda")

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config
)

window_size = 512
three_quarters = 384
half = 256
quarter = 128


def find_bottom(xywh):
    x, y, width = (xywh[0], xywh[1] + xywh[3], xywh[2])
    search_list = []

    for i in range((width // half) - 1):
        search_list.append((x + i * half, y - three_quarters))

    if width % half != 0:
        search_list.append((x + width - window_size, y - three_quarters))

    return search_list


def find_top(xywh):
    x, y, width = (xywh[0], xywh[1], xywh[2])
    search_list = []

    for i in range((width // half) - 1):
        search_list.append((x + i * half, y - quarter))

    if width % half != 0:
        search_list.append((x + width - window_size, y - quarter))

    return search_list


def find_left(xywh):
    x, y, height = (xywh[0], xywh[1], xywh[3])
    search_list = []

    for i in range((height // half) - 1):
        search_list.append((x - quarter, y + i * half))

    if height % half != 0:
        search_list.append((x - quarter, y + height - window_size))

    return search_list


def find_right(xywh):
    x, y, height = (xywh[0] + xywh[2], xywh[1], xywh[3])
    search_list = []

    for i in range((height // half) - 1):
        search_list.append((x - three_quarters, y + i * half))

    if height % half != 0:
        search_list.append((x - three_quarters, y + height - window_size))

    return search_list


def fill_bottom(xywh, image, mask_image):
    search_list = find_bottom(xywh)
    image, mask_image = fill_search_list(search_list, image, mask_image)

    xywh[3] += quarter

    return xywh, image, mask_image


def fill_left(xywh, image, mask_image):
    search_list = find_left(xywh)
    image, mask_image = fill_search_list(search_list, image, mask_image)

    xywh[0] -= quarter
    xywh[2] += quarter

    return xywh, image, mask_image


def fill_right(xywh, image, mask_image):
    search_list = find_right(xywh)
    image, mask_image = fill_search_list(search_list, image, mask_image)

    xywh[2] += quarter

    return xywh, image, mask_image


def fill_top(xywh, image, mask_image):
    search_list = find_top(xywh)
    image, mask_image = fill_search_list(search_list, image, mask_image)

    xywh[1] -= quarter
    xywh[3] += quarter

    return xywh, image, mask_image


def fill_search_list(search_list, image, mask_image):
    for x, y in search_list:
        # Create new mask
        new_mask_image = Image.new("RGB", mask_image.size, (255, 255, 255))
        mask_image_np = np.array(new_mask_image)

        # Fill the mask
        mask_image_np[y:y+window_size, x:x+window_size] = (0, 0, 0)

        # Mask to PIL
        new_mask_image = Image.fromarray(mask_image_np)

        mask2_gray = new_mask_image.convert("L")
        mask2_array = np.array(mask2_gray)

        mask1_gray = mask_image.convert("L")
        mask1_array = np.array(mask1_gray)

        # Generate and merge image
        generated_image = crop_image(
            image, mask_image, x, y, x+window_size, y+window_size)
        image = merge_image(image, generated_image, mask1_gray, x, y)

        # Update mask_image
        result_array = np.logical_and(mask1_array == 255, mask2_array == 255)
        mask_image = Image.fromarray(np.uint8(result_array)*255)

    return image, mask_image


def crop_image(image, result_mask, x, y, m, n):

    cropped_image = image.crop((x, y, m, n))
    cropped_mask = result_mask.crop((x, y, m, n))

    images = pipe(prompt="", image=cropped_image,
                  mask_image=cropped_mask, height=m-x, width=n-y, num_inference_steps=50).images

    return images[0]


def merge_image(image, generated_image, mask2_gray, x, y):
    mask = mask2_gray.crop(
        (x, y, x + generated_image.width, y + generated_image.height))
    image.paste(generated_image, (x, y), mask)
    return image


def calculate_buffer(num):
    remainder = num % 128
    if remainder == 0:
        return 0
    else:
        return 128 - remainder


def outpaint(image, left_border, top_border, right_border, bottom_border):
    image = Image.fromarray(image)
    mask_image = Image.new("RGB", image.size, (0, 0, 0))

    borders = [int(i)
               for i in [left_border, top_border, right_border, bottom_border]]
    buffers = [calculate_buffer(border) for border in borders]
    borders = [i + j for i, j in zip(borders, buffers)]

    w, h = image.size

    padded_borders = (borders[0], borders[1], borders[2], borders[3])

    image = ImageOps.expand(image, padded_borders, fill="white")
    mask_image = ImageOps.expand(mask_image, padded_borders, fill="white")

    xywh = [borders[0], borders[1], w, h]

    counts = [(x // 128) for x in borders]

    functions = [fill_left, fill_top, fill_right, fill_bottom]

    while any(counts):
        for i, func in enumerate(functions):
            if counts[i] > 0:
                xywh, image, mask_image = func(
                    xywh, image, mask_image)
                counts[i] -= 1

    width, height = image.size
    return image.crop((buffers[0], buffers[1], width - buffers[2], height - buffers[3]))
