from diffusers import StableDiffusionInpaintPipeline
from diffusers import EulerAncestralDiscreteScheduler
from PIL import Image, ImageOps
import numpy as np
import torch

# pipe = StableDiffusionInpaintPipeline.from_single_file(
#     "./civit_ai/dreamshaper_8Inpainting.safetensors",
#     torch_dtype=torch.float16,
# ).to('cuda')

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
).to('cuda')

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config
)

window_size = 512
three_quarters = 384
half = 256
quarter = 128


def find_bottom(left_bottom, width):
    x, y = left_bottom
    search_list = []

    for i in range((width // half) - 1):
        search_list.append((x + i * half, y - three_quarters))

    if width % half != 0:
        search_list.append((x + width - window_size, y - three_quarters))

    return search_list


def find_top(left_top, width):
    x, y = left_top
    search_list = []

    for i in range((width // half) - 1):
        search_list.append((x + i * half, y - quarter))

    if width % half != 0:
        search_list.append((x + width - window_size, y - quarter))

    return search_list


def find_left(left_top, height):
    x, y = left_top
    search_list = []

    for i in range((height // half) - 1):
        search_list.append((x - quarter, y + i * half))

    if height % half != 0:
        search_list.append((x - quarter, y + height - window_size))

    return search_list


def find_right(right_top, height):
    x, y = right_top
    search_list = []

    for i in range((height // half) - 1):
        search_list.append((x - three_quarters, y + i * half))

    if height % half != 0:
        search_list.append((x - three_quarters, y + height - window_size))

    return search_list


def fill_bottom(left_bottom, right_bottom, image, mask_image):
    search_list = find_bottom(left_bottom, right_bottom[0] - left_bottom[0])
    image, mask_image = fill_search_list(search_list, image, mask_image)

    left_bottom[1] += quarter
    right_bottom[1] += quarter

    return left_bottom, right_bottom, image, mask_image


def fill_left(left_top, left_bottom, image, mask_image):
    search_list = find_left(left_top, left_bottom[1] - left_top[1])
    image, mask_image = fill_search_list(search_list, image, mask_image)

    left_bottom[0] -= quarter
    left_top[0] -= quarter

    return left_top, left_bottom, image, mask_image


def fill_right(right_bottom, right_top, image, mask_image):
    search_list = find_right(right_top, right_bottom[1] - right_top[1])
    image, mask_image = fill_search_list(search_list, image, mask_image)

    right_top[0] += quarter
    right_bottom[0] += quarter

    return right_bottom, right_top, image, mask_image


def fill_top(left_top, right_top, image, mask_image):
    search_list = find_top(left_top, right_top[0] - left_top[0])
    image, mask_image = fill_search_list(search_list, image, mask_image)

    left_top[1] -= quarter
    right_top[1] -= quarter

    return left_top, right_top, image, mask_image


def fill_search_list(search_list, image, mask_image):
    for x, y in search_list:
        # Create new mask
        new_mask_image = Image.new('RGB', mask_image.size, (255, 255, 255))
        mask_image_np = np.array(new_mask_image)

        # Fill the mask
        mask_image_np[y:y+512, x:x+512] = (0, 0, 0)

        # Mask to PIL
        new_mask_image = Image.fromarray(mask_image_np)

        mask2_gray = new_mask_image.convert('L')
        mask2_array = np.array(mask2_gray)

        mask1_gray = mask_image.convert('L')
        mask1_array = np.array(mask1_gray)

        # Generate and merge image
        generated_image = crop_image(image, mask_image, x, y, x + 512, y + 512)
        image = merge_image(image, generated_image, mask1_gray, x, y)

        # Update mask_image
        result_array = np.logical_and(mask1_array == 255, mask2_array == 255)
        mask_image = Image.fromarray(np.uint8(result_array)*255)

    return image, mask_image


def crop_image(image, result_mask, x, y, m, n):

    print(x, y, m, n)

    cropped_image = image.crop((x, y, m, n))
    cropped_mask = result_mask.crop((x, y, m, n))

    images = pipe(prompt='', image=cropped_image,
                  mask_image=cropped_mask, height=m-x, width=n-y, num_inference_steps=50).images

    return images[0]


def merge_image(image, generated_image, mask2_gray, x, y):
    mask2_gray.save('mask2_gray.png')
    mask = mask2_gray.crop(
        (x, y, x + generated_image.width, y + generated_image.height))
    image.paste(generated_image, (x, y), mask)
    # image.paste(generated_image, (x, y))
    return image
