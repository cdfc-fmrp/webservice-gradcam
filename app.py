print("1. Iniciando app.py")

import os
import re

from flask import Flask, render_template, request
from PIL import Image
from datetime import datetime

from pipeline import (
    carregar_modelo,
    inicializar_modelo,
    analisar_imagem,
    gerar_gradcam_visual
)

print("2. Imports concluídos")

weights_path = "model.pth"

print("3. Carregando modelo...")
model = carregar_modelo(weights_path)

print("4. Modelo carregado")

model, device, gradcam = inicializar_modelo(model)

print("5. Modelo inicializado")

app = Flask(__name__)

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

print("6. Flask criado")


def identificar_imagem(nome_arquivo):

    paciente = "Não identificado"
    vertebra = "Não identificada"

    paciente_match = re.search(
        r"(P\d+)",
        nome_arquivo,
        re.IGNORECASE
    )

    vertebra_match = re.search(
        r"(L[1-5])",
        nome_arquivo,
        re.IGNORECASE
    )

    if paciente_match:
        paciente = paciente_match.group(1).upper()

    if vertebra_match:
        vertebra = vertebra_match.group(1).upper()

    return paciente, vertebra


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/historico")
def historico():
    return render_template("historico.html")


@app.route("/modelo")
def modelo():
    return render_template("modelo.html")


@app.route("/sobre")
def sobre():
    return render_template("sobre.html")


@app.route("/analisar", methods=["GET", "POST"])
def analisar():

    if request.method == "POST":

        arquivos = request.files.getlist("imagens")

        if not arquivos or arquivos[0].filename == "":
            return "Nenhuma imagem selecionada"

        classes_reais = request.form.getlist("classe_real")

        resultados = []

        data_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        nome_pasta = f"analise_{data_hora}"

        caminho_pasta = os.path.join(
            STATIC_DIR,
            "resultados",
            nome_pasta
        )

        os.makedirs(
            caminho_pasta,
            exist_ok=True
)

        for indice, arquivo in enumerate(arquivos):

            if arquivo.filename == "":
                continue

            nome_arquivo = arquivo.filename

            image = Image.open(arquivo).convert("RGB")

            # Paciente e vértebra identificados pelo nome
            paciente, vertebra = identificar_imagem(
                nome_arquivo
            )

            # Mesmo pipeline que já funcionava
            resultado, cam, image_processada = analisar_imagem(
                image,
                model,
                device,
                gradcam
            )

            # Classe real informada no formulário
            classe_real = None

            if indice < len(classes_reais):

                valor = classes_reais[indice].strip().upper()

                if valor in ["BENIGN", "MALIGNANT"]:
                    classe_real = valor

            # Classe prevista pelo modelo
            classe_prevista = resultado["classe"]

            # Verifica se acertou
            if classe_real:
                acertou = classe_prevista == classe_real
            else:
                acertou = None

            # Numeração das imagens dentro da análise
            numero_imagem = indice + 1

            nome_imagem_original = (
                f"{numero_imagem:02d}_original.png"
            )

            nome_imagem_gradcam = (
                f"{numero_imagem:02d}_gradcam.png"
            )

            caminho_imagem_original = os.path.join(
                caminho_pasta,
                nome_imagem_original
            )

            caminho_imagem_gradcam = os.path.join(
                caminho_pasta,
                nome_imagem_gradcam
            )

            image.save(caminho_imagem_original)

            gerar_gradcam_visual(
                image_processada,
                cam,
                resultado["classe"],
                caminho_imagem_gradcam
            )

            url_imagem_original = os.path.join(
                "resultados",
                nome_pasta,
                nome_imagem_original
            ).replace("\\", "/")

            url_imagem_gradcam = os.path.join(
                "resultados",
                nome_pasta,
                nome_imagem_gradcam
            ).replace("\\", "/")

            resultados.append({
                "nome_arquivo": nome_arquivo,
                "paciente": paciente,
                "vertebra": vertebra,
                "classe_real": classe_real,
                "classe_prevista": classe_prevista,
                "acertou": acertou,
                "resultado": resultado,
                "imagem_original": url_imagem_original,
                "imagem_gradcam": url_imagem_gradcam
            })

        return render_template(
            "resultado.html",
            resultados=resultados
        )

    return render_template("analisar.html")


if __name__ == "__main__":

    print("7. Iniciando servidor Flask")

    app.run(
        debug=True,
        use_reloader=False
    )