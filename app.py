from flask import Flask, request, jsonify
from deepface import DeepFace
import pickle
import numpy as np

# Carrega embeddings
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

model_name = "Facenet"
THRESHOLD = 0.6  # ajuste conforme DeepFace

app = Flask(__name__)

@app.route("/reconhecer", methods=["POST"])
def reconhecer():
    if "imagem" not in request.files:
        return jsonify({"erro": "Nenhuma imagem enviada"}), 400

    file = request.files["imagem"]

    try:
        embedding = DeepFace.represent(img_path=file,
                                       model_name=model_name,
                                       enforce_detection=False)

        if embedding:
            vetor = embedding[0]["embedding"]

            menor_dist = float("inf")
            pessoa = "Desconhecido"

            for nome, emb in embeddings.items():
                dist = np.linalg.norm(np.array(vetor) - np.array(emb))
                if dist < menor_dist:
                    menor_dist = dist
                    pessoa = nome

            if menor_dist < THRESHOLD:
                return jsonify({"pessoa": pessoa, "distancia": menor_dist})
            else:
                return jsonify({"pessoa": "Desconhecido", "distancia": menor_dist})

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run()