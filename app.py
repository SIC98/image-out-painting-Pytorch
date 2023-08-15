import pytorch_lightning as pl
import gradio as gr

from repainting import sepia
from out_painting import image_mod

pl.seed_everything(42)


with gr.Blocks() as demo:
    gr.Markdown("Out-painting & Re-painting demo")
    with gr.Tab("Out painting"):
        with gr.Row():
            first_img = gr.Image()
            outpaint_img = gr.Image()
        left_border = gr.Textbox(label="left border")
        right_border = gr.Textbox(label="right border")
        top_border = gr.Textbox(label="top border")
        bottom_border = gr.Textbox(label="bottom border")
        outpaint_button = gr.Button("Paint")
    with gr.Tab("Repainting"):
        with gr.Row():
            second_img = gr.Image()
            replace_img = gr.Image()
            target_img = gr.Image()
        repaint_button = gr.Button("Paint")

    outpaint_button.click(image_mod, inputs=[first_img, left_border, top_border, right_border, bottom_border],
                          outputs=outpaint_img)
    repaint_button.click(sepia, inputs=second_img,
                         outputs=[replace_img, target_img])

demo.launch()
