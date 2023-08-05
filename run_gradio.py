from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageOps
import gradio as gr
import torch
import numpy as np

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    # "runwayml/stable-diffusion-inpainting",
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
).to('cuda')

def fill_bottom(left_bottom, width):
    x, y = left_bottom
    search_list = []

    for i in range((width // 256) - 1):
        search_list.append((x + i * 256, y - 384))

    if width % 256 != 0:
        search_list.append((x + width - 512, y - 384))

    print('search list:', search_list)
    return search_list


def fill_top(left_top, width):
    x, y = left_top
    search_list = []

    for i in range((width // 256) - 1):
        search_list.append((x + i * 256, y - 128))

    if width % 256 != 0:
        search_list.append((x + width - 512, y - 128))

    print('search list:', search_list)
    return search_list

def fill_left(left_top, height):
    x, y = left_top
    search_list = []

    for i in range((height // 256) - 1):
        search_list.append((x - 128, y + i * 256))

    if height % 256 != 0:
        search_list.append((x - 128, y + height - 512))
    
    return search_list

def fill_right(right_top, height):
    x, y = right_top
    search_list = []

    for i in range((height // 256) - 1):
        search_list.append((x - 384, y + i * 256))

    if height % 256 != 0:
        search_list.append((x - 384, y + height - 512))
    
    return search_list


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

            # Generate and merge image
            generated_image = crop_image(image, mask_image, x, y, x + 512, y + 512)
            image = merge_image(image, generated_image, x, y)

            mask1_gray = mask_image.convert('L')
            mask1_array = np.array(mask1_gray)

            # Update mask_image
            result_array = np.logical_and(mask1_array == 255, mask2_array == 255)
            mask_image = Image.fromarray(np.uint8(result_array)*255)

    return image, mask_image


def crop_image(image, result_mask, x, y, m, n):

    print(x, y, m, n)

    cropped_image = image.crop((x, y, m, n))
    cropped_mask = result_mask.crop((x, y, m, n))

    image = pipe(prompt='', image=cropped_image, mask_image=cropped_mask, height=m-x, width=n-y).images[0]

    return image


def merge_image(image, generated_image, x, y):
    image.paste(generated_image, (x, y))
    return image


def posible_start_point(image):
    x, y = image.size
    return max(0, x - 512), max(0, y - 512)



def image_mod(image, left_border, right_border, top_border, bottom_border):
    mask_image = Image.new('RGB', image.size, (0, 0, 0))

    left_border, right_border, top_border, bottom_border = int(left_border), int(right_border), int(top_border), int(bottom_border)

    print(image.size)
    w, h = image.size
    result_w = w + left_border + right_border
    result_h = h + top_border + bottom_border
    add_w = -result_w % 8
    add_h = -result_h % 8

    border = (left_border, top_border, right_border + add_w, bottom_border + add_h)

    image = ImageOps.expand(image, border, fill='white')
    mask_image = ImageOps.expand(mask_image, border, fill='white')

    # search_list = find_left(left_border, top_border, right_border + add_w, bottom_border + add_h)

    left_top, left_bottom, right_top, right_bottom = [left_border, top_border], [left_border, top_border + h], [left_border + w, top_border], [left_border + w, top_border + h]

    search_list = fill_bottom(left_bottom, right_bottom[0] - left_bottom[0])
    image, mask_image = fill_search_list(search_list, image, mask_image)

    left_bottom[1] += 128
    right_bottom[1] += 128

    search_list = fill_left(left_top, left_bottom[1] - left_top[1])
    image, mask_image = fill_search_list(search_list, image, mask_image)

    left_bottom[0] -= 128
    left_top[0] -= 128

    search_list = fill_top(left_top, right_bottom[0] - left_bottom[0])
    image, mask_image = fill_search_list(search_list, image, mask_image)

    left_top[1] -= 128
    right_top[1] -= 128

    search_list = fill_right(right_top, left_bottom[1] - left_top[1])
    image, mask_image = fill_search_list(search_list, image, mask_image)

    right_top[0] += 128
    right_bottom[0] += 128


    return image




    # x, y = posible_start_point(image)

    # new_mask_image = Image.fromarray(mask_image_np)
    # mask2_gray = new_mask_image.convert('L')

    # mask1_array = np.array(mask1_gray)
    # mask2_array = np.array(mask2_gray)

    # result_array = np.logical_or(mask1_array == 255, mask2_array == 255)

    # generated_image = crop_image(image, mask_image, 0, 0, 512, 512)
    # image = merge_image(image, generated_image, 0, 0)

    # Convert the result back to an image
    # mask_image = Image.fromarray(np.uint8(result_array)*255)

    # image = pipe(prompt='', image=image, mask_image=result_mask, height=height, width=width).images[0]

    # return image
    
    width, height = image.size
    return image.crop((0, 0, width - add_w, height - add_h))


# demo = gr.Interface(
#     fn=image_mod,
#     inputs=[gr.Image(type="pil"), 'text', 'text', 'text', 'text'],
#     outputs="image",
#     # flagging_options=["blurry", "incorrect", "other"],
# )

if __name__ == "__main__":
    # demo.launch()
    image = Image.open('./imges/image1.png')
    image = image.resize((1024,1024))
    image = image_mod(image, 512, 512, 512, 512)
    image.save("./new_cube.png")