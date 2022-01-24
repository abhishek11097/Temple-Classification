# Temple Classification
The repository contains the prediction code for a classifier that could guess which country the temple is in. The code gets the path to a directory with images as a parameter, and returns a CSV file with the results.
The solution has option of three pretrained models: Efficient Net B3, Efficient B0, VGG16. Efficient Net B3 is the default model since it gives the highest accuracy.

## Setup
### Requirements
|library|version|
|----|----|
|numpy|1.22.1|
|opencv-python|4.5.5.62|
|pandas|1.4.0|
|Pillow|9.0.0|
|python-dateutil|2.8.2|
|pytz|2021.3|
|six|1.16.0|
|torch|1.10.1|
|torchvision|0.11.2|
|typing_extensions|4.0.1|

### Steps for installing requirements
`pip3 install -r requirements.txt`

## Usage
`bash predict.sh --input_path [PATH TO IMAGE DIRECTORY] --model [MODEL NAME] --pretrained_path [Path of the Pretrained Model] --output_dir [Path to the directory where output will be written] --image_shape [Dimension of the input shape]`
### Arguments
|Argument|Required/Optional|Meaning|
|----|----|----|
|--input_path|Required|Path of the Input Directory
|--model|Optional [Default: EFFB3]|Prediction Model to be used
|--pretrained_path|Optional [Default: models/temple-classifier-eff-best.pt]|Path of the pretrained model
|--output_dir|Optional [Default: data/output/]|Path of the Output Directory
|--image_shape|Optional [Default: 300]|Width and Height of the image

### Examples
* `bash predict.sh --input_path data/input/`
* `bash predict.sh --input_path data/input/ --model EFFB0 --pretrained_path models/temple-classifier-effb0.pt --image_shape 224`
* `bash predict.sh --input_path data/input/ --model VGG16 --pretrained_path models/temple-classifier-vgg.pt --image_shape 224`

> Paste the pretrained models downloaded from the link below in the models file

### Helper for the bash script
`bash predict.sh --help`

### Training Process of prediction model
The images were divided in training and test dataset on a 80:20 split. Data augmentation was performed on training data.

The following augmentation techniques were applied:
* Random 90 Degree Rotation
* Random Crop
* Adding Gaussian Noise
* Adding Fog like noise
* Changing the color temperature(to give a night likeview)
* Redacting Random Parts of image 

<p align="center">
  <img src="data_aug.png">
</p>

### Results of Test Set
|Model|Accuracy|Weights|
|----|----|----|
|EfficientNet B3|84.61|[effb3weights](https://drive.google.com/file/d/12rduB0SrQSS3QgVoPfuKmhANwjXsxNF1/view?usp=sharing)|
|EfficientNet B0|81.24|[effb0weights](https://drive.google.com/file/d/1KaN8nNyp5RJiy2LDZIoP0Q_sgMjSPODC/view?usp=sharing)|
|VGG16|79.84|[vggweights](https://drive.google.com/file/d/1I-PIIunZenf_jo-u3Nhwk6v24nk1ZX-A/view?usp=sharing)|
