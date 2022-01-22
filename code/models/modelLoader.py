import logging 
import os

from models.vgg16 import vgg16ModelHandler
from exceptionHandler import raiseException

def loadPredictionModel(args, logger):
    selected_model = args.model
    number_of_classes = 11
    device = "cpu"
    pretrained_weight_path = args.pretrained_path
    if(selected_model == "VGG16"):
        try:
            model = vgg16ModelHandler(pretrained_weight_path, number_of_classes, device, logger)
        except:
            raiseException("Unable to create Model class")

    else:
        logger.exception("Invalid Model Choice")
        raise raiseException(logger)

    try:
        model.loadModelWeights()
    except Exception as e:
        logger.exception("Unable to load Model Weights")
        raise raiseException("Unable to load Model Weights")


    return None