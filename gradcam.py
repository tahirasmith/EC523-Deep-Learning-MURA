import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor):
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output.squeeze()
        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def load_model(path):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 1)

    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return image, tensor


def overlay(image, cam):
    image = image.resize((224, 224))
    image = np.array(image)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    overlay = heatmap + np.float32(image) / 255
    overlay = overlay / overlay.max()

    return np.uint8(255 * overlay)


if __name__ == "__main__":
    model = load_model("densenet_best.pth")

    target_layer = model.features.denseblock4

    gradcam = GradCAM(model, target_layer)
    
    ## get path of png orig image and and add here for diff images
    image_path = "MURA-v1.1/train/XR_FOREARM/patient09148/study1_positive/image1.png" 

    image, tensor = preprocess(image_path)
    image.save("original.png")

    # prediction
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()
        print(f"Prediction probability: {prob:.4f}")

    # grad-cam
    cam = gradcam.generate(tensor)
    result = overlay(image, cam)

    cv2.imwrite("gradcam_result.png", result)
    print("Saved gradcam_result.png")
