import torch.nn as nn
import torch
from torchvision import models


def get_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    # replace classifier for binary task
    model.classifier = nn.Linear(model.classifier.in_features, 1)

    # IMPORTANT: disable in-place ops for Grad-CAM stability
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    return model
