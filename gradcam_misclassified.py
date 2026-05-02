import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from model_densenet import get_model


# -------------------------
# Grad-CAM
# -------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x):
        self.model.zero_grad()

        # CRITICAL FIX: avoid view/in-place conflict
        x = x.clone()

        output = self.model(x)
        loss = output.sum()
        loss.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam


# -------------------------
# Overlay
# -------------------------
def overlay(image, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    image = np.float32(image) / 255

    result = heatmap + image
    result = result / result.max()

    return np.uint8(255 * result)


# -------------------------
# Main
# -------------------------
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model()
    model.load_state_dict(torch.load("densenet_best.pth", map_location=device))
    model.to(device)
    model.eval()

    # 🔥 CRITICAL FIX: disable in-place ReLU everywhere
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    target_layer = model.features[-1]

    gradcam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    input_dirs = [
        "misclassified/false_pos",
        "misclassified/false_neg"
    ]

    os.makedirs("gradcam_outputs", exist_ok=True)

    count = 0

    for folder in input_dirs:
        if not os.path.exists(folder):
            continue

        for fname in os.listdir(folder):
            if not fname.endswith(".png"):
                continue

            path = os.path.join(folder, fname)

            image = Image.open(path).convert("RGB")
            image_np = np.array(image.resize((224, 224)))

            tensor = transform(image).unsqueeze(0).to(device)

            cam = gradcam.generate(tensor)
            result = overlay(image_np, cam)

            save_path = os.path.join("gradcam_outputs", f"cam_{fname}")
            cv2.imwrite(save_path, result)

            count += 1

    print(f"Saved {count} Grad-CAM images to ./gradcam_outputs/")


if __name__ == "__main__":
    run()
