import pytorch_lightning as pl
from PIL import Image, ImageOps
import gradio as gr

from utils import fill_bottom, fill_top, fill_left, fill_right

pl.seed_everything(42)


def image_mod(image, left_border, right_border, top_border, bottom_border):
    mask_image = Image.new('RGB', image.size, (0, 0, 0))

    left_border, right_border, top_border, bottom_border = int(
        left_border), int(right_border), int(top_border), int(bottom_border)

    print(image.size)
    w, h = image.size
    result_w = w + left_border + right_border
    result_h = h + top_border + bottom_border
    add_w = -result_w % 8
    add_h = -result_h % 8

    border = (left_border, top_border, right_border +
              add_w, bottom_border + add_h)

    image = ImageOps.expand(image, border, fill='white')
    mask_image = ImageOps.expand(mask_image, border, fill='white')

    XYWH = [left_border, top_border, w, h]

    for _ in range(0, 2):
        XYWH, image, mask_image = fill_bottom(XYWH, image, mask_image)
        XYWH, image, mask_image = fill_left(XYWH, image, mask_image)
        XYWH, image, mask_image = fill_top(XYWH, image, mask_image)
        XYWH, image, mask_image = fill_right(XYWH, image, mask_image)

    # for _ in range(0, 2):
    #     left_top, left_bottom, image, mask_image = fill_left(
    #         left_top, left_bottom, image, mask_image)
    #     right_bottom, right_top, image, mask_image = fill_right(
    #         right_bottom, right_top, image, mask_image)

    width, height = image.size
    return image.crop((0, 0, width - add_w, height - add_h))


# demo = gr.Interface(
#     fn=image_mod,
#     inputs=[gr.Image(type="pil"), 'text', 'text', 'text', 'text'],
#     outputs="image",
#     # flagging_options=["blurry", "incorrect", "other"],
# )
# demo.launch()

if __name__ == '__main__':
    image = Image.open('./images/bus.jpeg')
    print(image.size)
    image = image_mod(image, 256, 256, 256, 256)
    image.save('./results/bus.jpeg')
