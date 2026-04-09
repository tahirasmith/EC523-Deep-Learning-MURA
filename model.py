import torch.nn as nn
import torchvision.models as models


def get_model(model_name="resnet18", pretrained=True):
    """
    Returns a model for binary classification (normal vs abnormal)
    """

    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, 1)

    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, 1)

    else:
        raise ValueError(f"Model {model_name} not supported")

    return model
