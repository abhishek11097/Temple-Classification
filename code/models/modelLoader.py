import logging 
import os

from models.vgg16 import vgg16ModelHandler
from exceptionHandler import raiseException

import configparser

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
    return model


def getModelPrediction(temple_prediction_model, processed_image_tensor_list,logger):
    output_prediction = []
    for processed_image_tensor in processed_image_tensor_list:
        predicted_class = temple_prediction_model.predictModel(processed_image_tensor.processed_image)
        output_prediction.append({"ImagePath":processed_image_tensor.input_image_path,"Prediction":predicted_class})
    return output_prediction


