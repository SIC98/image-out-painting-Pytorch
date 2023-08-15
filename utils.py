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


def find_bottom(left_bottom, width):
    x, y = left_bottom
    search_list = []

    for i in range((width // 256) - 1):
        search_list.append((x + i * 256, y - 384))

    if width % 256 != 0:
        search_list.append((x + width - 512, y - 384))

    return search_list


def find_top(left_top, width):
    x, y = left_top
    search_list = []

    for i in range((width // 256) - 1):
        search_list.append((x + i * 256, y - 128))

    if width % 256 != 0:
        search_list.append((x + width - 512, y - 128))

    return search_list


def find_left(left_top, height):
    x, y = left_top
    search_list = []

    for i in range((height // 256) - 1):
        search_list.append((x - 128, y + i * 256))

    if height % 256 != 0:
        search_list.append((x - 128, y + height - 512))

    return search_list


def find_right(right_top, height):
    x, y = right_top
    search_list = []

    for i in range((height // 256) - 1):
        search_list.append((x - 384, y + i * 256))

    if height % 256 != 0:
        search_list.append((x - 384, y + height - 512))

    return search_list


def fill_bottom(left_bottom, right_bottom, image, mask_image):
    search_list = find_bottom(left_bottom, right_bottom[0] - left_bottom[0])
    image, mask_image = fill_search_list(search_list, image, mask_image)

    left_bottom[1] += 128
    right_bottom[1] += 128

    return left_bottom, right_bottom, image, mask_image


def fill_left(left_top, left_bottom, image, mask_image):
    search_list = find_left(left_top, left_bottom[1] - left_top[1])
    image, mask_image = fill_search_list(search_list, image, mask_image)

    left_bottom[0] -= 128
    left_top[0] -= 128

    return left_top, left_bottom, image, mask_image


def fill_right(right_bottom, right_top, image, mask_image):
    search_list = find_right(right_top, right_bottom[1] - right_top[1])
    image, mask_image = fill_search_list(search_list, image, mask_image)

    right_top[0] += 128
    right_bottom[0] += 128

    return right_bottom, right_top, image, mask_image


def fill_top(left_top, right_top, image, mask_image):
    search_list = find_top(left_top, right_top[0] - left_top[0])
    image, mask_image = fill_search_list(search_list, image, mask_image)

    left_top[1] -= 128
    right_top[1] -= 128

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
