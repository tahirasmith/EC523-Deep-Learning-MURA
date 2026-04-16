import torch.nn as nn
import torchvision.models as models


def get_model():
    model = models.resnet18(pretrained=True)

    # binary classification
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model
