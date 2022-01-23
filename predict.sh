#!/bin/bash
source processArgs.sh "$@"

python3 code/templePredictor.py --input_path $input_path --model $model --pretrained_path $pretrained_path --output_dir $output_dir --image_shape $image_shape