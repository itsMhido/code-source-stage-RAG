import faiss, pickle, numpy as np, argparse
from sentence_transformers import SentenceTransformer

INDEX_PATH  = "fos_index.faiss"
META_PATH   = "fos_metadata.pkl"
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K       = 5

def load_resources():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metas = pickle.load(f)
    model = SentenceTransformer(MODEL_NAME)
    return index, metas, model

def search(query, index, metas, model, k=TOP_K):
    q_vec = model.encode([query]).astype("float32")
    scores, idxs = index.search(q_vec, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        results.append({"score": float(score), **metas[idx]})
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Votre question")
    args = parser.parse_args()

    index, metas, model = load_resources()
    hits = search(args.query, index, metas, model)

    print("\n===== TOP K RESULTATS =====")
    for h in hits:
        print(f"[score={h['score']:.2f}] {h['text']}")
