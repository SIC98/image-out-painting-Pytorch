from PIL import Image
from run_gradio import image_mod
import os

# directory/folder path
directory_path = 'coco2014/val2014/'
out_painted_path = 'coco2014/out_painted/'

# list to store files
all_files_and_dirs = os.listdir(directory_path)


for item in all_files_and_dirs[:10]:
    full_path = os.path.join(directory_path, item)
    if os.path.isfile(full_path):

        image = Image.open(full_path)
        print(image.size)
        image = image_mod(image, 256, 256, 256, 256)
        image.save(
            os.path.join(out_painted_path, item)
        )
