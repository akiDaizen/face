from deepface import DeepFace
import os
import pickle
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

db_path = "base/"
embeddings = {}

# Modelo a ser usado
model_name = "Facenet"

print("Gerando embeddings da base...")

for file in os.listdir(db_path):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        nome = os.path.splitext(file)[0]
        caminho = os.path.join(db_path, file)

        try:
            embedding = DeepFace.represent(img_path=caminho,
                                           model_name=model_name,
                                           enforce_detection=False)
            embeddings[nome] = embedding[0]["embedding"]
            print(f"[OK] Embedding gerado para {nome}")

        except Exception as e:
            print(f"[ERRO] {file} - {e}")

# Salva embeddings em arquivo
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Embeddings salvos em embeddings.pkl")