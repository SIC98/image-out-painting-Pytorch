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
- The process of painting a 1024 x 1024 image can be visualized as follows.

| Step 0                                                                                                             | Step 1                                                                                                             | Step 2                                                                                                             | Step 7                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| ![step0](https://github.com/SIC98/image-out-painting-Pytorch/assets/51232785/5918afdb-e5f1-4190-a063-caeeca3228d7) | ![step1](https://github.com/SIC98/image-out-painting-Pytorch/assets/51232785/d4f1f2aa-d65d-455f-a553-8b68ca3ec78a) | ![step2](https://github.com/SIC98/image-out-painting-Pytorch/assets/51232785/1dbf77de-d5ee-4c77-ae6f-6f2e58927465) | ![step7](https://github.com/SIC98/image-out-painting-Pytorch/assets/51232785/87ddcfd5-9bdb-4510-babe-ebaec2cf84f5) |

