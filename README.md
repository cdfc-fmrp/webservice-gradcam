# SpineVision

Aplicação web para **classificação de imagens de ressonância magnética da região lombar utilizando Inteligência Artificial**, com geração de mapas de explicabilidade por meio de **Grad-CAM**.

O sistema realiza a classificação de imagens utilizando um modelo baseado em **VGG16** e apresenta as predições, as probabilidades associadas e visualizações das regiões que mais contribuíram para as decisões do modelo.

## Instalação via Docker — V3

A versão 3 do SpineVision está disponível como imagem Docker através do GitHub Container Registry (GHCR).

A utilização do Docker elimina a necessidade de instalar Python, configurar ambientes virtuais ou baixar os pesos do modelo separadamente.

### Pré-requisitos

- Docker instalado e em execução;
- Computador com arquitetura x86_64 (AMD64).

### 1. Baixar a imagem Docker

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

Caso o pacote seja privado, será necessário possuir
permissão de acesso e autenticar-se previamente:

```bash
docker login ghcr.io
```

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

Remover o container e os dados armazenados
exclusivamente nele:

```bash
docker rm -f spinevision
```

## Versões do projeto

| Versão | Código-fonte | Imagem Docker |
|---|---|---|
| Original | [main](https://github.com/cdfc-fmrp/webservice-gradcam/tree/main) | — |
| V2 | [v2](https://github.com/cdfc-fmrp/webservice-gradcam/tree/v2) | — |
| V3 | [v3](https://github.com/cdfc-fmrp/webservice-gradcam/tree/v3) | `ghcr.io/cdfc-fmrp/spinevision:v3` |

A documentação completa da versão 3 está disponível
na [branch v3](https://github.com/cdfc-fmrp/webservice-gradcam/tree/v3).

As informações abaixo descrevem a estrutura e a
arquitetura da versão original do projeto.

## Funcionalidades

- Upload de imagens nos formatos JPG, JPEG e PNG;
- Classificação entre as classes `BENIGN` e `MALIGNANT`;
- Exibição da confiança da predição;
- Probabilidade individual de cada classe;
- Geração de visualização Grad-CAM;
- Interface web desenvolvida com Flask;
- Disponibilização do serviço em uma instância AWS EC2.

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

### Infraestrutura e Distribuição

- AWS EC2
- Gunicorn
- Nginx
- Amazon S3
- Docker (V3)
- GitHub Container Registry (V3)

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

O arquivo `model.pth` não é versionado diretamente
no GitHub devido ao seu tamanho, mas está incorporado
à imagem Docker da V3.

## Arquitetura da aplicação

O projeto foi dividido em duas principais camadas.

### `pipeline.py`

Responsável pelo processamento relacionado ao modelo
de Inteligência Artificial:

- Pré-processamento da imagem;
- Carregamento do modelo;
- Classificação;
- Cálculo das probabilidades;
- Geração do Grad-CAM;
- Criação da visualização final.

### `app.py`

Responsável pela aplicação web:

- Gerenciamento das rotas Flask;
- Recebimento das imagens;
- Comunicação com o pipeline de IA;
- Geração dos resultados;
- Renderização das páginas HTML.

## Execução tradicional

Para execução sem Docker, é necessário configurar
o ambiente Python, instalar as dependências e
disponibilizar os pesos do modelo.

Com o ambiente virtual ativado:

```bash
source venv/bin/activate
```

O serviço pode ser iniciado utilizando Gunicorn:

```bash
gunicorn --bind 127.0.0.1:8000 app:app
```

Na infraestrutura utilizada no projeto, o **Nginx**
recebe as requisições HTTP e as encaminha para
o Gunicorn, que executa a aplicação Flask.

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

O modelo utilizado é baseado na arquitetura **VGG16**,
adaptada para a classificação das imagens em duas classes:

- `BENIGN`
- `MALIGNANT`

Após a classificação, o sistema utiliza **Grad-CAM
(Gradient-weighted Class Activation Mapping)** para
destacar as regiões da imagem que apresentaram maior
contribuição para a decisão do modelo.

## Sobre o Grad-CAM

O Grad-CAM utiliza os gradientes da classe prevista
em relação às ativações de uma camada convolucional
da rede.

No SpineVision, essa técnica é utilizada como ferramenta
de **explicabilidade do modelo**, auxiliando na
visualização das regiões relevantes para a predição.

As regiões destacadas não devem ser interpretadas
diretamente como localização de uma lesão ou
diagnóstico clínico.

## Origem do projeto

O SpineVision foi desenvolvido como parte de atividades
envolvendo **Inteligência Artificial, processamento de
imagens médicas e desenvolvimento de software**,
integrando um modelo de Deep Learning a uma aplicação
web disponibilizada em infraestrutura AWS.

A lógica inicialmente explorada em notebooks foi
posteriormente reorganizada em um pipeline reutilizável
para integração com o WebService.

## Aviso

Este projeto possui caráter **acadêmico e experimental**.

Os resultados fornecidos pelo modelo não constituem
diagnóstico médico e não substituem a avaliação de
profissionais da saúde.

Utilize apenas imagens anonimizadas e autorizadas
para processamento.