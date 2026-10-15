import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


labels = ["BENIGN", "MALIGNANT"]


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocessar_imagem(image, device):
    input_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )

    return input_tensor


def classificar_imagem(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)

    predicted_class = probabilities.argmax(dim=1).item()
    
    confianca = probabilities[0, predicted_class].item()

    return {
        "classe": labels[predicted_class],
        "class_idx": predicted_class,
        "confianca": confianca,
        "prob_benign": probabilities[0, 0].item(),
        "prob_malignant": probabilities[0, 1].item()
    }


class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_hook = self.target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, inputs, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):

        self.model.zero_grad()

        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (weights * activations).sum(dim=1)

        cam = torch.relu(cam)

        cam = (
            cam.squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        cam = cam - cam.min()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, class_idx

def carregar_modelo(weights_path):
    checkpoint = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=False
    )

    return checkpoint

def inicializar_modelo(model):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    model.eval()

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    target_layer = model.features[28]

    gradcam = GradCAM(
        model,
        target_layer
    )

    return model, device, gradcam


def analisar_imagem(image, model, device, gradcam):

    input_tensor = preprocessar_imagem(
        image,
        device
    )

    resultado = classificar_imagem(
        model,
        input_tensor
    )

    cam, class_idx = gradcam(
        input_tensor,
        class_idx=resultado["class_idx"]
    )

    return resultado, cam

def gerar_gradcam_visual(image, cam, classe, caminho_saida):
    cam_resized = Image.fromarray(
        (cam * 255).astype(np.uint8)
    ).resize(
        (image.width, image.height),
        resample=Image.BILINEAR
    )

    cam_resized = np.array(cam_resized) / 255.0

    plt.figure(figsize=(7, 7))

    plt.imshow(image)
    plt.imshow(
        cam_resized,
        cmap="jet",
        alpha=0.45
    )

    plt.axis("off")
    plt.title(f"Grad-CAM - {classe}")

    plt.savefig(
        caminho_saida,
        bbox_inches="tight",
        dpi=200
    )

    plt.close()