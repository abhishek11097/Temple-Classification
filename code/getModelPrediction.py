import os
import configparser
import argparse
import logging

from models.modelLoader import loadPredictionModel
from exceptionHandler import raiseException

if(not os.path.isdir(os.path.join(os.getcwd(),"../logs/"))):
    os.mkdir(os.path.join(os.getcwd(),"../logs/"))

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] [%(process)d]  %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

LOG_FILE_PATH = os.path.join(os.getcwd(), "../logs/templePrediction.log")
fileHandler = logging.FileHandler(LOG_FILE_PATH)
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler) 

logger.info("Loading Config File")
config = configparser.RawConfigParser()
config.read(os.path.join(os.getcwd(),"../config/config.property"))

DEFAULT_VALUES = dict(config.items("DEFAULT"))
DEFAULT_MODEL = DEFAULT_VALUES["model"]
DEFAULT_PRETRAINED_MODEL_PATH = os.path.join(os.getcwd(),DEFAULT_VALUES["pretrained_model_path"])
DEFAULT_OUTPUT_DIRECTORY = os.path.join(os.getcwd(),DEFAULT_VALUES["output_directory"])
DEFAULT_INPUT_SHAPE = int(DEFAULT_VALUES["input_shape"])

def parseArguments():
    logger.info("Loading User Arguments")
    parser = argparse.ArgumentParser(description='PyTorch Semantic Video Segmentation training')
    parser.add_argument('--input_path', type=str, required=True, help="Path to the directory containing input images")
    parser.add_argument('--model', type=str, default = DEFAULT_MODEL, help = "Model to be used for prediction; Default: VGG16")
    parser.add_argument('--pretrained_path', type=str, default = DEFAULT_PRETRAINED_MODEL_PATH, help="Path of the pretrained Model; Default: ../models/temple-classifier-best.pt")
    parser.add_argument('--output_dir', type=str, default = DEFAULT_OUTPUT_DIRECTORY, help="Path where output csv is shared; Default: ../data/output/")
    parser.add_argument('--image_shape', type=int, default = DEFAULT_INPUT_SHAPE, help="Shape of Input Image; Default is 224")
    args = parser.parse_args()
    logger.info("Input Path: " + args.input_path)
    logger.info("Evaluation Model: " + args.model)
    logger.info("Pretrained Model Path: " + args.pretrained_path)
    logger.info("Output Directory: " + args.output_dir)
    logger.info("Size of Input Image: 3*" + str(args.image_shape)+"*"+str(args.image_shape))
    return args

def main():
    try:
        args = parseArguments()
    except:
        logger.exception("Exception while loading user arguments")
        raise raiseException("Exception while loading user arguments")

    temple_prediction_model = loadPredictionModel(args, logger)

    # predicted_output = getModelPrediction(temple_prediction_model, )
    # output_file_path = writePredictionOutput(predicted_output)
    return None

if __name__ == "__main__":
    main()