import json, os, pickle, uuid, argparse, re
from pathlib import Path
from tqdm import tqdm

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------- Paramètres ----------
JSON_PATH   = "structured_qa_dataset.json"
INDEX_PATH  = "fos_index.faiss"
META_PATH   = "fos_metadata.pkl"
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE  = 500      # en tokens (~≈ 400 mots)
CHUNK_OVERLAP = 50     # pour garder le contexte d’un chunk sur l’autre
# --------------------------------

def load_qa(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)

def pre_clean(text):
    """Nettoyage minimal : espaces, balises multiples, etc."""
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", "،", "؛", " "],
    )
    all_chunks, metas = [], []
    for doc in docs:
        # Concat question/response dans un même doc, ou traite-les séparément : ici on concat
        full_text = f"{doc['type'].upper()} : {doc['content']}"
        for chunk in splitter.split_text(full_text):
            chunk_id = str(uuid.uuid4())
            all_chunks.append(chunk)
            metas.append({
                "id": chunk_id,
                "type": doc["type"],
                "ppr": doc.get("ppr", ""),
                "source_doc": doc,   # garde le doc entier pour une citation éventuelle
                "text": chunk
            })
    return all_chunks, metas

def build_index(chunks, model_name, index_path, meta_path):
    print("→ Chargement du modèle d’embedding …")
    model = SentenceTransformer(model_name)

    print("→ Calcul des embeddings …")
    embeddings = model.encode(chunks, batch_size=64, show_progress_bar=True, convert_to_numpy=True).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    print("→ Ajout des vecteurs à l’index FAISS …")
    index.add(embeddings)

    print(f"→ Sauvegarde de l’index dans {index_path}")
    faiss.write_index(index, index_path)

    print(f"→ Sauvegarde des métadonnées dans {meta_path}")
    with open(meta_path, "wb") as f:
        pickle.dump(metas, f)

    print("✅ Indexation terminée !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=JSON_PATH)
    args = parser.parse_args()

    print("→ Lecture du dataset …")
    documents = load_qa(args.json)

    print("→ Chunking des documents …")
    chunks, metas = chunk_documents(documents)

    build_index(chunks, MODEL_NAME, INDEX_PATH, META_PATH)
