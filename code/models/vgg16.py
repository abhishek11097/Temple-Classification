import torch
from torchvision import models
import torch.nn as nn

CLASS_LABEL_LIST = []

class vgg16ModelHandler:
    def __init__(self, pretrained_weight_path, number_of_classes, device, logger):
        self.number_of_classes = number_of_classes
        self.device = device
        self.pretrained_weight_path = pretrained_weight_path
        self.logger = logger
        self.logger.info("Creating Model")
        self.model = models.vgg16()
        self.logger.info("Model Created Successfully")

    def loadModelWeights(self):
        self.logger.info("Loading Model Weights from file " + self.pretrained_weight_path)
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, 11)
        model_path = torch.load(self.pretrained_weight_path,map_location=torch.device(self.device))
        self.model.load_state_dict(model_path)
        self.logger.info("Successfully Loaded Model")


    def getClassName(self, predicted_label):
        return CLASS_LABEL_LIST[predicted_label]

    def getModelPrediction(self, dataloader):
        self.model.eval()
        output_prediction = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted_labels = torch.max(outputs, 1)

                predicted_class = [self.getClassName(predicted_label) for predicted_label in predicted_labels]
                output_prediction += predicted_class
        return output_prediction