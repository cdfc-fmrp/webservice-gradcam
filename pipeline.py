import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from PIL import Image


labels = ["BENIGN", "MALIGNANT"]

transform_visual = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])
transform_modelo = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocessar_imagem(image, device):

    image_processada = transform_visual(image)

    input_tensor = (
        transform_modelo(image_processada)
        .unsqueeze(0)
        .to(device)
    )

    return input_tensor, image_processada


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

def gerar_gradiente_saliencia(
    model,
    input_tensor,
    class_idx,
    corte_percentil=99
):
    # Cria uma cópia porque precisamos calcular
    # o gradiente em relação à entrada
    x = (
        input_tensor
        .clone()
        .detach()
        .requires_grad_(True)
    )

    model.zero_grad(set_to_none=True)

    output = model(x)

    # Logit correspondente à classe prevista
    score = output[:, class_idx]

    score.backward()

    # Gradiente absoluto máximo entre os 3 canais
    mapa = (
        x.grad
        .abs()
        .amax(dim=1)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    # Limita valores extremos
    limite = np.percentile(
        mapa,
        corte_percentil
    )

    mapa = np.minimum(
        mapa,
        limite
    )

    # Normalização 0–1
    mapa = mapa - mapa.min()

    if mapa.max() > 0:
        mapa = mapa / mapa.max()

    return mapa.astype(np.float32)

def gerar_gradiente_visual(
    image,
    mapa,
    classe,
    caminho_saida
):
    imagem_np = np.array(image)

    intensidade = imagem_np.mean(axis=2)

    mascara_vertebra = intensidade > 10

    plt.figure(figsize=(7, 7))

    plt.imshow(image)

    plt.imshow(
        mapa,
        cmap="jet",
        alpha=0.45
    )

    plt.contour(
        mascara_vertebra.astype(float),
        levels=[0.5],
        colors="white",
        linewidths=1.5
    )

    plt.axis("off")

    plt.savefig(
        caminho_saida,
        bbox_inches="tight",
        dpi=200
    )

    plt.close()

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

        self.model.zero_grad(set_to_none=True)

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
    input_tensor, image_processada = preprocessar_imagem(
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

    saliencia = gerar_gradiente_saliencia(
        model,
        input_tensor,
        resultado["class_idx"]
    )

    return (
        resultado,
        cam,
        saliencia,
        image_processada
    )

def gerar_gradcam_visual(image, cam, classe, caminho_saida):
    cam_resized = Image.fromarray(
        (cam * 255).astype(np.uint8)
    ).resize(
        (image.width, image.height),
        resample=Image.BILINEAR
    )

    cam_resized = np.array(cam_resized) / 255.0

    imagem_np = np.array(image)

    intensidade = imagem_np.mean(axis=2)

    mascara_vertebra = intensidade > 10

    plt.figure(figsize=(7, 7))

    plt.imshow(image)
    plt.imshow(
        cam_resized,
        cmap="jet",
        alpha=0.45
    )

    plt.contour(
        mascara_vertebra.astype(float),
        levels=[0.5],
        colors="white",
        linewidths=1.5
    )

    plt.axis("off")

    plt.savefig(
        caminho_saida,
        bbox_inches="tight",
        dpi=200
    )

    plt.close()