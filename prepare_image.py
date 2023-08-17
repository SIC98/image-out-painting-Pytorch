from PIL import Image
import os

from out_painting import outpaint

coco_data_path = "coco2014/val2014/"
out_painted_path = "coco2014/out_painted/"

# list to store files
all_files_and_dirs = os.listdir(coco_data_path)


for item in all_files_and_dirs[:100]:
    full_path = os.path.join(coco_data_path, item)
    if os.path.isfile(full_path):

        image = Image.open(full_path)
        print(image.size)
        image = outpaint(image, 256, 256, 256, 256)
        image.save(
            os.path.join(out_painted_path, item)
        )
