import pytorch_lightning as pl
from PIL import Image, ImageOps
import gradio as gr

from utils import fill_bottom, fill_top, fill_left, fill_right
from sam_gradio import sepia

pl.seed_everything(42)


def calculate_buffer(num):
    remainder = num % 128
    if remainder == 0:
        return 0
    else:
        return 128 - remainder


def image_mod(image, left_border, top_border, right_border, bottom_border):
    image = Image.fromarray(image)
    mask_image = Image.new('RGB', image.size, (0, 0, 0))

    borders = [int(i)
               for i in [left_border, top_border, right_border, bottom_border]]
    buffers = [calculate_buffer(border) for border in borders]
    borders = [i + j for i, j in zip(borders, buffers)]

    w, h = image.size

    padded_borders = (borders[0], borders[1], borders[2], borders[3])

    image = ImageOps.expand(image, padded_borders, fill='white')
    mask_image = ImageOps.expand(mask_image, padded_borders, fill='white')

    XYWH = [borders[0], borders[1], w, h]

    counts = [(x // 128) for x in borders]

    functions = [fill_left, fill_top, fill_right, fill_bottom]

    while any(counts):
        print(counts)
        for i, func in enumerate(functions):
            if counts[i] > 0:
                XYWH, image, mask_image = func(
                    XYWH, image, mask_image)
                counts[i] -= 1

    width, height = image.size
    return image.crop((buffers[0], buffers[1], width - buffers[2], height - buffers[3]))


# demo = gr.Interface(
#     fn=image_mod,
#     inputs=[gr.Image(type="pil"), 'text', 'text', 'text', 'text'],
#     outputs="image",
#     # flagging_options=["blurry", "incorrect", "other"],
# )
# demo.launch()

if __name__ == '__main__':

    with gr.Blocks() as demo:
        gr.Markdown("Flip text or image files using this demo.")
        with gr.Tab("Flip Text"):
            with gr.Row():
                input_img = gr.Image()
                replace_img = gr.Image()
                target_img = gr.Image()
            repaint_button = gr.Button("repaint")
        with gr.Tab("Out painting"):
            with gr.Row():
                input_img = gr.Image()
                outpaint_img = gr.Image()
            left_border = gr.Textbox(label="left border")
            right_border = gr.Textbox(label="right border")
            top_border = gr.Textbox(label="top border")
            bottom_border = gr.Textbox(label="bottom border")
            outpaint_button = gr.Button("Paint")

        repaint_button.click(sepia, inputs=input_img,
                             outputs=[replace_img, target_img])
        outpaint_button.click(image_mod, inputs=[input_img, left_border, top_border, right_border, bottom_border],
                              outputs=outpaint_img)

    demo.launch()
