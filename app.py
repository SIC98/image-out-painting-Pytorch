import pytorch_lightning as pl
from PIL import Image, ImageOps
import gradio as gr

from sam_gradio import sepia
from out_painting import image_mod

pl.seed_everything(42)


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
