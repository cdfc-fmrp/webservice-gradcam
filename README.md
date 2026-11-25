# SpineVision

O **SpineVision** é um web service desenvolvido para classificação de imagens segmentadas de vértebras lombares utilizando Deep Learning.

A aplicação permite que o usuário envie uma imagem e receba como resultado a classificação entre:

- **BENIGN**
- **MALIGNANT**

Além da classificação, o sistema apresenta a probabilidade associada a cada classe e utiliza **Grad-CAM** para auxiliar na interpretação da decisão do modelo.

## Modelo

O modelo utilizado é baseado na arquitetura **VGG16**, implementada com PyTorch.

Durante a inferência, as imagens passam pelas seguintes etapas de pré-processamento:

- Resize
- Center Crop para 224 × 224 pixels
- Conversão para tensor
- Normalização utilizando os valores do ImageNet

## Grad-CAM

O SpineVision utiliza Grad-CAM para gerar um mapa visual das regiões da imagem que mais contribuíram para a classificação realizada pelo modelo.

O mapa é sobreposto à imagem original, permitindo observar as regiões consideradas mais relevantes para a decisão.

## Tecnologias utilizadas

- Python
- PyTorch
- Torchvision
- Flask
- NumPy
- Pillow
- Matplotlib
- HTML
- CSS

## Estrutura do projeto

```text
webservice-gradcam/
│
├── app.py
├── pipeline.py
├── model.pth
├── requirements.txt
├── Dockerfile
│
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── analisar.html
│   ├── resultado.html
│   ├── historico.html
│   ├── modelo.html
│   └── sobre.html
│
└── static/
    └── style.css
```

## Fluxo da aplicação

```text
Upload da imagem
        ↓
Pré-processamento
        ↓
Modelo VGG16
        ↓
Classificação
        ↓
Grad-CAM
        ↓
Resultado na interface web
```

## Objetivo

O projeto foi desenvolvido com foco em explorar a integração entre modelos de Deep Learning para imagens médicas e aplicações web, permitindo disponibilizar a inferência do modelo por meio de uma interface simples.

## Aviso

O SpineVision foi desenvolvido para fins de pesquisa, desenvolvimento e demonstração tecnológica.

O sistema não deve ser utilizado isoladamente para diagnóstico ou tomada de decisão clínica.