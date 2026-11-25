# SpineVision

Web service desenvolvido para classificação de imagens segmentadas de vértebras lombares utilizando Deep Learning e técnicas de Inteligência Artificial Explicável (XAI).

O sistema recebe imagens, realiza a classificação entre **BENIGN** e **MALIGNANT** e apresenta recursos visuais para auxiliar na interpretação da decisão do modelo.

---

## Funcionalidades

- Upload de múltiplas imagens
- Classificação automática entre **BENIGN** e **MALIGNANT**
- Probabilidade estimada para cada classe
- Identificação automática do paciente e da vértebra a partir do nome do arquivo
- Comparação opcional com a classe real
- Visualização por **Grad-CAM**
- Visualização por **Gradiente de Saliência**
- Contorno da vértebra segmentada sobre os mapas de explicabilidade
- Armazenamento das imagens resultantes por análise
- Página com informações sobre o modelo e desempenho
- Dashboard interativo da base utilizada no desenvolvimento do modelo
- Execução containerizada com Docker

---

## Modelo

O SpineVision utiliza uma arquitetura **VGG16** para classificação de imagens de vértebras lombares segmentadas.

As classes utilizadas são:

```text
BENIGN
MALIGNANT
```

As imagens são processadas antes da inferência utilizando:

- Resize para 256 pixels
- Center Crop para 224 × 224
- Conversão para tensor
- Normalização utilizando os valores do ImageNet

O modelo recebe, portanto, imagens de entrada com dimensão efetiva de **224 × 224 pixels**.

---

## Base de dados

A base utilizada possui:

| Classe | Imagens |
|---|---:|
| BENIGN | 417 |
| MALIGNANT | 367 |
| **Total** | **784** |

A divisão utilizada foi:

| Conjunto | Imagens |
|---|---:|
| Treino | 627 |
| Validação | 73 |
| Teste | 84 |

A separação foi realizada em nível de paciente, mantendo todas as imagens pertencentes ao mesmo paciente no mesmo conjunto e evitando sobreposição entre treino, validação e teste.

---

## Desempenho

Resultados obtidos no conjunto de teste:

| Métrica | Resultado |
|---|---:|
| Acurácia | 83,33% |
| Sensibilidade — MALIGNANT | 85,37% |
| Especificidade | 81,40% |
| ROC AUC | 0,94 |

### Matriz de confusão

| | Predito BENIGN | Predito MALIGNANT |
|---|---:|---:|
| Real BENIGN | 35 | 8 |
| Real MALIGNANT | 6 | 35 |

O conjunto de teste contém **84 imagens**, sendo:

- 43 BENIGN
- 41 MALIGNANT

---

## Inteligência Artificial Explicável — XAI

Além da classificação, o SpineVision utiliza técnicas de explicabilidade para visualizar regiões relacionadas à decisão do modelo.

### Grad-CAM

O Grad-CAM utiliza as ativações e gradientes da última camada convolucional da VGG16 para gerar um mapa espacial das regiões que mais contribuíram para a classe analisada.

No SpineVision, o Grad-CAM é calculado utilizando:

```python
model.features[28]
```

O mapa resultante é redimensionado e sobreposto à imagem utilizada pelo modelo.

### Gradiente de Saliência

O Gradiente de Saliência calcula o gradiente da saída da classe prevista em relação aos pixels da imagem de entrada.

Isso permite visualizar quais regiões da imagem apresentam maior sensibilidade em relação à predição.

O mapa utiliza o valor absoluto máximo dos gradientes entre os três canais da imagem.

### Contorno da vértebra

Nos dois mapas de explicabilidade, o limite da vértebra segmentada é representado por um contorno branco.

Esse contorno é utilizado apenas como referência visual e não modifica os cálculos de Grad-CAM ou Gradiente de Saliência.

---

## Dashboard da base

A página **Modelo** contém um dashboard interativo com informações sobre a base utilizada no projeto.

Entre as informações disponíveis estão:

- distribuição por sexo;
- distribuição entre BENIGN e MALIGNANT;
- idade dos pacientes;
- quantidade de vértebras por paciente;
- distribuição das vértebras lombares;
- distribuição das imagens entre treino, validação e teste;
- distribuição das classes por vértebra.

O dashboard permite aplicar filtros para explorar diferentes grupos da base.

---

# Execução

## Forma recomendada — imagem Docker `.tar`

A forma principal de executar o SpineVision é utilizando a imagem Docker previamente construída e disponibilizada em formato `.tar`.

Essa abordagem mantém o ambiente, as dependências e o modelo utilizados na versão validada da aplicação.

### 1. Carregar a imagem

Com Docker instalado:

```bash
docker load -i spinevision.tar
```

Verifique se a imagem foi carregada:

```bash
docker images
```

### 2. Executar o container

```bash
docker run --name spinevision -p 8080:8080 spinevision:latest
```

A aplicação estará disponível em:

```text
http://localhost:8080
```

Para interromper:

```bash
docker stop spinevision
```

Para iniciar novamente:

```bash
docker start spinevision
```

---

## Construção a partir do código-fonte

O código-fonte também está disponível neste repositório.

Caso seja necessário gerar uma nova imagem:

```bash
docker build -t spinevision:latest .
```

Depois execute:

```bash
docker run --name spinevision -p 8080:8080 spinevision:latest
```

---

## Estrutura principal

```text
webservice-gradcam/
│
├── app.py
├── pipeline.py
├── model.pth
├── Dockerfile
├── requirements.txt
│
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── analisar.html
│   ├── resultado.html
│   ├── historico.html
│   ├── modelo.html
│   ├── dashboard_base.html
│   └── sobre.html
│
└── static/
    ├── style.css
    └── resultados/
```

---

## Principais tecnologias

- Python
- PyTorch
- Torchvision
- Flask
- NumPy
- Pillow
- Matplotlib
- HTML
- CSS
- JavaScript
- Chart.js
- Docker

---

## Fluxo da aplicação

```text
Upload das imagens
        ↓
Pré-processamento
        ↓
VGG16
        ↓
Classificação BENIGN / MALIGNANT
        ↓
Probabilidades
        ↓
Gradiente de Saliência
        ↓
Grad-CAM
        ↓
Geração das visualizações
        ↓
Interface web
```

---

## Resultados gerados

Cada análise cria uma nova pasta em:

```text
static/resultados/
```

Exemplo:

```text
static/
└── resultados/
    └── analise_2026-11-25_15-30-00/
        ├── 01_original.png
        ├── 01_gradiente.png
        ├── 01_gradcam.png
        ├── 02_original.png
        ├── 02_gradiente.png
        └── 02_gradcam.png
```

---

## Observação

O valor apresentado como **Confiança** corresponde à probabilidade obtida após a aplicação de Softmax sobre a saída do modelo.

Esse valor representa o score do modelo para a classificação realizada e não deve ser interpretado isoladamente como uma medida de certeza clínica.

---

## Aviso

O SpineVision foi desenvolvido para fins de pesquisa, desenvolvimento e demonstração tecnológica.

O sistema não deve ser utilizado isoladamente para diagnóstico ou tomada de decisão clínica.