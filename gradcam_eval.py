import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model_densenet import get_model
from dataset import get_dataloaders


# -------------------------
# Grad-CAM implementation
# -------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # hooks
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x):
        self.model.zero_grad()

        # IMPORTANT: keep graph for Grad-CAM
        x = x.requires_grad_(True)

        output = self.model(x)

        score = output.squeeze()

        score.backward()

        pooled_grads = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        cam = torch.sum(pooled_grads * self.activations, dim=1).squeeze()

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), output


# -------------------------
# Image utilities
# -------------------------
def denormalize(img):
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    return img


def overlay_cam(img, cam):
    cam = np.uint8(255 * cam)
    cam = np.stack([cam] * 3, axis=-1)

    cam = cam.astype(np.float32) / 255.0
    img = img.astype(np.float32)

    return np.clip(0.5 * img + 0.5 * cam, 0, 1)


def save_fig(original, cam, overlay, idx, out_dir):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(original)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(cam, cmap="jet")
    ax[1].set_title("Grad-CAM")
    ax[1].axis("off")

    ax[2].imshow(overlay)
    ax[2].set_title("Overlay")
    ax[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"gradcam_{idx}.png"))
    plt.close()


# -------------------------
# Main evaluation
# -------------------------
def run(num_samples=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = get_dataloaders(batch_size=1)

    model = get_model().to(device)
    model.load_state_dict(torch.load("densenet_best.pth", map_location=device))

    model.eval()

    # FIX: disable inplace ReLU (prevents Grad-CAM crash)
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False

    # target layer (DenseNet last conv block)
    target_layer = model.features[-1]

    gradcam = GradCAM(model, target_layer)

    os.makedirs("gradcam_results", exist_ok=True)

    count = 0

    for img, label, body_part in val_loader:

        if count >= num_samples:
            break

        img = img.to(device)

        cam, output = gradcam.generate(img)

        prob = torch.sigmoid(output).item()
        pred = 1 if prob > 0.5 else 0

        original = denormalize(img.squeeze(0))
        overlay = overlay_cam(original, cam)

        save_fig(original, cam, overlay, count, "gradcam_results")

        print(f"[{count}] Label={label.item()} Pred={pred} Prob={prob:.3f} {body_part[0]}")

        count += 1


if __name__ == "__main__":
    run()
