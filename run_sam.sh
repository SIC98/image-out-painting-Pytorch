#!/bin/bash

CHECKPOINT_PATH="sam_vit_h_4b8939.pth"
MODEL_TYPE="vit_h"
OUTPUT_PATH="coco2014/mask/"
INPUT_FOLDER="coco2014/out_painted/"

for FILE in $INPUT_FOLDER*; do
    if [ -f "$FILE" ]; then
        python segment-anything/scripts/amg.py --checkpoint $CHECKPOINT_PATH --model-type $MODEL_TYPE --input $FILE --output $OUTPUT_PATH
    fi
done