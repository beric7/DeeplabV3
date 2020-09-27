""" DeepLabv3 Model download and change the head for your prediction"""
from models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
# n_classes is the number of probabilities you want to get per pixel
#   - For 1 class and background, use n_classes=1
#   - For 2 classes, use n_classes=1
#   - For N > 2 classes, use n_classes=N

def createDeepLabv3(outputchannels=3):
    model = models.segmentation.deeplabv3_resnet101(
        pretrained=True, progress=True)
    # Added a Tanh activation after the last convolution layer
    # What does 2048 as the input channel mean? 
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model
