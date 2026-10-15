print("1. Iniciando app.py")

import os
import uuid
from flask import Flask, render_template, request
from PIL import Image
from pipeline import (
    carregar_modelo,
    inicializar_modelo,
    analisar_imagem,
    gerar_gradcam_visual
)

print("2. Imports concluídos")

weights_path = "/home/ec2-user/SageMaker/vgg16_model/model.pth"

print("3. Carregando modelo...")
model = carregar_modelo(weights_path)

print("4. Modelo carregado")

model, device, gradcam = inicializar_modelo(model)

print("5. Modelo inicializado")

app = Flask(__name__)

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

print("6. Flask criado")


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        if "imagem" not in request.files:
            return "Nenhuma imagem enviada"

        arquivo = request.files["imagem"]

        if arquivo.filename == "":
            return "Nenhuma imagem selecionada"

        image = Image.open(arquivo).convert("RGB")

        print("Imagem aberta.")
        print("Chamando pipeline...")

        resultado, cam = analisar_imagem(
            image,
            model,
            device,
            gradcam
        )

        print("Pipeline finalizado.")
        print("Imagem recebida:", arquivo.filename)
        print("Classe:", resultado["classe"])
        print("BENIGN:", resultado["prob_benign"])
        print("MALIGNANT:", resultado["prob_malignant"])
        print("Grad-CAM:", cam.shape)

        identificador = uuid.uuid4().hex

        nome_imagem_original = f"imagem_original_{identificador}.png"
        nome_imagem_gradcam = f"gradcam_resultado_{identificador}.png"

        caminho_imagem_original = os.path.join(
            STATIC_DIR,
            nome_imagem_original
        )

        caminho_imagem_gradcam = os.path.join(
            STATIC_DIR,
            nome_imagem_gradcam
        )

        image.save(caminho_imagem_original)

        gerar_gradcam_visual(
            image,
            cam,
            resultado["classe"],
            caminho_imagem_gradcam
        )

        return render_template(
            "index.html",
            resultado=resultado,
            imagem_original=nome_imagem_original,
            imagem_gradcam=nome_imagem_gradcam
        )

    return render_template("index.html")

if __name__ == "__main__":
    print("7. Iniciando servidor Flask")
    app.run(debug=True, use_reloader=False)