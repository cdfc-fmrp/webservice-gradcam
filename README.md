# SpineVision

Aplicação web para **classificação de imagens de ressonância magnética da região lombar utilizando Inteligência Artificial**, com geração de mapas de explicabilidade por meio de **Grad-CAM**.

O sistema permite o envio de uma imagem pela interface web, realiza a classificação utilizando um modelo baseado em **VGG16** e apresenta a predição, as probabilidades associadas e uma visualização das regiões que mais contribuíram para a decisão do modelo.

## Funcionalidades

- Upload de imagens nos formatos JPG, JPEG e PNG
- Classificação entre as classes `BENIGN` e `MALIGNANT`
- Exibição da confiança da predição
- Probabilidade individual de cada classe
- Geração de visualização **Grad-CAM**
- Interface web desenvolvida com Flask
- Disponibilização do serviço em uma instância AWS EC2

## Fluxo da aplicação

```text
Usuário
   ↓
Nginx
   ↓
Gunicorn
   ↓
Flask
   ↓
Pipeline de IA
   ↓
VGG16
   ↓
Classificação + Grad-CAM
   ↓
Resultado exibido na interface
```

## Tecnologias

### Inteligência Artificial
- Python
- PyTorch
- Torchvision
- VGG16
- Grad-CAM
- NumPy
- Pillow

### Aplicação Web
- Flask
- HTML
- CSS
- Jinja
- Bootstrap Icons

### Infraestrutura
- AWS EC2
- Gunicorn
- Nginx
- Amazon S3

## Estrutura do projeto

```text
webservice-gradcam/
│
├── app.py
├── pipeline.py
│
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── analisar.html
│   ├── resultado.html
│   ├── historico.html
│   └── sobre.html
│
├── static/
│   └── style.css
│
├── Classificacao.ipynb
├── GRAD-CAM.ipynb
├── metrica-grafico.ipynb
│
├── .gitignore
└── README.md
```

## Arquitetura da aplicação

O projeto foi dividido em duas principais camadas.

### `pipeline.py`

Responsável pelo processamento relacionado ao modelo de Inteligência Artificial:

- pré-processamento da imagem;
- carregamento do modelo;
- classificação;
- cálculo das probabilidades;
- geração do Grad-CAM;
- criação da visualização final.

### `app.py`

Responsável pela aplicação web:

- gerenciamento das rotas Flask;
- recebimento das imagens;
- comunicação com o pipeline de IA;
- geração dos resultados;
- renderização das páginas HTML.

## Execução

Com o ambiente virtual ativado:

```bash
source venv/bin/activate
```

O serviço pode ser iniciado utilizando Gunicorn:

```bash
gunicorn --bind 127.0.0.1:8000 app:app
```

Na infraestrutura utilizada no projeto, o **Nginx** recebe as requisições HTTP e as encaminha para o Gunicorn, que executa a aplicação Flask.

```text
Internet
   ↓
Nginx :80
   ↓
Gunicorn 127.0.0.1:8000
   ↓
Flask
   ↓
Modelo de IA
```

## Modelo

O modelo utilizado é baseado na arquitetura **VGG16**, adaptada para a classificação das imagens em duas classes:

```text
BENIGN
MALIGNANT
```

Após a classificação, o sistema utiliza **Grad-CAM (Gradient-weighted Class Activation Mapping)** para destacar as regiões da imagem que apresentaram maior contribuição para a decisão do modelo.

## Sobre o Grad-CAM

O Grad-CAM utiliza os gradientes da classe prevista em relação às ativações de uma camada convolucional da rede.

No SpineVision, essa técnica é utilizada como ferramenta de **explicabilidade do modelo**, auxiliando na visualização das regiões relevantes para a predição.

As regiões destacadas não devem ser interpretadas diretamente como localização de uma lesão ou diagnóstico clínico.

## Origem do projeto

O SpineVision foi desenvolvido como parte de atividades envolvendo **Inteligência Artificial, processamento de imagens médicas e desenvolvimento de software**, integrando um modelo de Deep Learning a uma aplicação web disponibilizada em infraestrutura AWS.

A lógica inicialmente explorada em notebooks foi posteriormente reorganizada em um pipeline reutilizável para integração com o WebService.

## Aviso

Este projeto possui caráter **acadêmico e experimental**.

Os resultados fornecidos pelo modelo não constituem diagnóstico médico e não substituem a avaliação de profissionais da saúde.
