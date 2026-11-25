# SpineVision — V3

Aplicação web para **classificação de imagens de ressonância magnética da região lombar utilizando Inteligência Artificial**, com geração de mapas de explicabilidade por meio de **Grad-CAM**.

O sistema permite o envio de imagens pela interface web, realiza a classificação utilizando um modelo baseado em **VGG16** e apresenta as predições, probabilidades associadas e visualizações das regiões que mais contribuíram para as decisões do modelo.

## Instalação via Docker

O SpineVision pode ser executado utilizando Docker, sem a necessidade de instalar Python, configurar ambientes virtuais ou baixar os pesos do modelo separadamente.

### Pré-requisitos

- Docker instalado e em execução.
- Computador com arquitetura x86_64 (AMD64).

### 1. Baixar a imagem

```bash
docker pull ghcr.io/cdfc-fmrp/spinevision:v3
```

### 2. Executar o container

```bash
docker run -d --name spinevision -p 127.0.0.1:8080:8080 ghcr.io/cdfc-fmrp/spinevision:v3
```

### 3. Acessar a aplicação

Abra o navegador e acesse:

http://localhost:8080

A imagem Docker já contém:

- Código-fonte da aplicação;
- Dependências Python;
- PyTorch e Torchvision para CPU;
- Pesos do modelo VGG16;
- Interface web;
- Servidor Gunicorn.

Caso o pacote seja privado, será necessário realizar a autenticação no GitHub Container Registry com uma conta autorizada antes do download.

### Gerenciamento do container

Parar a aplicação:

```bash
docker stop spinevision
```

Iniciar novamente:

```bash
docker start spinevision
```

Consultar os logs:

```bash
docker logs spinevision
```

Remover o container e seus dados locais:

```bash
docker rm -f spinevision
```

## Funcionalidades

- Upload de imagens nos formatos JPG, JPEG e PNG;
- Análise individual ou de múltiplas imagens;
- Classificação entre as classes `BENIGN` e `MALIGNANT`;
- Exibição da confiança da predição;
- Probabilidade individual de cada classe;
- Geração de visualizações Grad-CAM;
- Interface web desenvolvida com Flask;
- Página de informações sobre o modelo;
- Execução em ambientes isolados utilizando Docker.

## Fluxo da aplicação

```text
Usuário
   ↓
Interface Web
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
Resultados exibidos na interface
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
- Matplotlib

### Aplicação Web

- Flask
- HTML
- CSS
- Jinja
- Bootstrap Icons

### Infraestrutura e Distribuição

- Docker
- GitHub Container Registry (GHCR)
- Gunicorn
- Nginx
- AWS EC2
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
│   ├── dashboard_base.html
│   ├── home.html
│   ├── analisar.html
│   ├── resultado.html
│   ├── modelo.html
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
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

O arquivo `model.pth` não é versionado diretamente
no GitHub devido ao seu tamanho, mas está incorporado
à imagem Docker distribuída.

## Arquitetura da aplicação

O projeto foi dividido em duas principais camadas.

### `pipeline.py`

Responsável pelo processamento relacionado ao modelo
de Inteligência Artificial:

- Pré-processamento das imagens;
- Carregamento do modelo;
- Classificação;
- Cálculo das probabilidades;
- Geração do Grad-CAM;
- Criação das visualizações finais.

### `app.py`

Responsável pela aplicação web:

- Gerenciamento das rotas Flask;
- Recebimento das imagens;
- Comunicação com o pipeline de IA;
- Organização dos resultados;
- Renderização das páginas HTML.

## Construção local da imagem Docker

Para desenvolvedores que desejam modificar o código
e construir sua própria imagem, é necessário possuir
o arquivo `model.pth` na raiz do projeto.

```bash
git clone -b v3 https://github.com/cdfc-fmrp/webservice-gradcam.git
cd webservice-gradcam
```

Após disponibilizar os pesos do modelo:

```bash
docker build -t spinevision:v3 .
```

Executar a imagem construída:

```bash
docker run -d --name spinevision-v3 -p 127.0.0.1:8080:8080 spinevision:v3
```

## Infraestrutura AWS

A aplicação também pode ser implantada em uma
instância AWS EC2, utilizando Nginx como proxy reverso
para encaminhar as requisições ao Gunicorn.

```text
Internet
   ↓
Nginx
   ↓
Gunicorn
   ↓
Flask
   ↓
Modelo de IA
```

## Modelo

O modelo utilizado é baseado na arquitetura **VGG16**,
adaptada para a classificação das imagens em duas classes:

- `BENIGN`
- `MALIGNANT`

Após a classificação, o sistema utiliza Grad-CAM
(Gradient-weighted Class Activation Mapping) para
destacar as regiões da imagem que apresentaram
maior contribuição para a decisão do modelo.

## Sobre o Grad-CAM

O Grad-CAM utiliza os gradientes da classe prevista
em relação às ativações de uma camada convolucional
da rede.

No SpineVision, essa técnica é utilizada como
ferramenta de **explicabilidade do modelo**,
auxiliando na visualização das regiões relevantes
para a predição.

As regiões destacadas não devem ser interpretadas
diretamente como localização de uma lesão ou
diagnóstico clínico.

## Origem do projeto

O SpineVision foi desenvolvido como parte de
atividades envolvendo Inteligência Artificial,
processamento de imagens médicas e desenvolvimento
de software, integrando um modelo de Deep Learning
a uma aplicação web.

A lógica inicialmente explorada em notebooks foi
posteriormente reorganizada em um pipeline
reutilizável para integração com o WebService.

## Aviso

Este projeto possui caráter **acadêmico e experimental**.

Os resultados fornecidos pelo modelo não constituem
diagnóstico médico e não substituem a avaliação de
profissionais da saúde.

Utilize apenas imagens anonimizadas e autorizadas
para processamento.