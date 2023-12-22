## image-out-painting-Pytorch

Gradio app for image out-painting and re-painting.

### Features

1. It can perform out-painting at any resolution.
   - It has improved the drawbacks of existing out-painting apps. For example, when using the `StableDiffusionInpaintPipeline` from the diffusers library, the width and height of the input image must be multiples of 8, and attempting to paint over excessively wide areas can result in an OOM (Out of Memory) error.
2. Automatically segments awkwardly generated out-painting areas and performs re-painting.

## Gradio app

| out-painting                                                                                                              | re-painting                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| ![out-painting](https://github.com/SIC98/image-out-painting-Pytorch/assets/51232785/a6e66483-725f-4fde-b02b-d2c5d13597b4) | ![re-painting](https://github.com/SIC98/image-out-painting-Pytorch/assets/51232785/059ffcad-156d-4b27-8366-b4d0f5b6c2ae) |

## Out-painting strategy

- From the Hugging Face Hub, I used the painting model [stabilityai/stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) and the Euler Ancestral scheduler.
- I employ a strategy of repeatedly out-painting up to a 512x512 area at a time to perform out-painting on large images.
- The process of painting a 2048 x 2048 image can be visualized as follows.

| Step 0                                                                                                             | Step 1                                                                                                             | Step 2                                                                                                             | Step 7                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| ![step0](https://github.com/SIC98/image-out-painting-Pytorch/assets/51232785/5918afdb-e5f1-4190-a063-caeeca3228d7) | ![step1](https://github.com/SIC98/image-out-painting-Pytorch/assets/51232785/d4f1f2aa-d65d-455f-a553-8b68ca3ec78a) | ![step2](https://github.com/SIC98/image-out-painting-Pytorch/assets/51232785/1dbf77de-d5ee-4c77-ae6f-6f2e58927465) | ![step7](https://github.com/SIC98/image-out-painting-Pytorch/assets/51232785/87ddcfd5-9bdb-4510-babe-ebaec2cf84f5) |


## Re-painting strategy

I trained the Resnet50 model to perform binary classification on awkward areas among those segmented by the SAM (Segment Anything) model.

- For the model's input, I combined the RGB 3-channel of the image with a mask channel to indicate the object's area, resulting in 4-channel data.
- Since there was no existing label data suitable for the classification I wanted, I performed the labeling myself and saved the values in `label.json`.
- I used 16 images to create a training dataset with 1,208 masks and a test dataset with 413 masks.

|          | Train | Test |
| -------- | ----- | ---- |
| Accuracy | 1.00  | 0.84 |

## Limitation

- The time it takes for out-painting increases significantly as the resolution of the image increases.
- More labeling data is required for higher reliability of the classification model.

## How to run my code

1. Install Python dependency & clone and install SAM repo.
```
pip install -r requirements.txt
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```
2. Download the resources needed for model training. The data has been saved in the following folder structure.

- coco2014/val2014: [Download link](http://images.cocodataset.org/zips/val2014.zip)
- sam_vit_h_4b8939.pth: [Download link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
```
.
├── coco2014
│   ├── val2014
│   │   ├── COCO_val2014_000000001987.jpg
│   │   ├── COCO_val2014_000000002764.jpg
│   │   └── ...
│   ├── mask
│   │   ├── COCO_val2014_000000001987
│   │   │   ├── 0.png
│   │   │   └── ...
│   │   ├── COCO_val2014_000000002764
│   │   │    ├── 0.png
│   │   │    └── ...
│   │   └── ...
│   ├── out_painted
│   │   ├── COCO_val2014_000000001987.jpg
│   │   ├── COCO_val2014_000000002764.jpg
│   │   └── ...
│   └──  
├── label.json
├── sam_vit_h_4b8939.pth
```
3. The data in the `coco2014/out_painted` and `coco2014/mask` folders can be obtained by executing the following.
```
python prepare_image.py
sh run_sam.sh
```
4. Train Resnet (SAM object classification) model.
```
python train.py
```
5. In `repainting.py`, specify the `ckpt_path` and run Gradio.
```
python app.py
```









