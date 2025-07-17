from flask import Flask, request, Response, send_from_directory
from ollama import chat
import faiss
import pickle
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load resources once
index = faiss.read_index("fos_index.faiss")
with open("fos_metadata.pkl", "rb") as f:
    metas = pickle.load(f)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chat_history = [
    {
        "role": "system",
        "content": "Tu es un assistant intelligent qui répond en se basant uniquement sur les documents fournis."
    }
]

def search_context(query, k=5):
    q_vec = embedder.encode([query]).astype("float32")
    scores, idxs = index.search(q_vec, k)
    return [metas[i]["text"] for i in idxs[0] if i < len(metas)]

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.json
    question = data.get("question", "")
    if not question:
        return {"error": "No question provided"}, 400

    context = "\n".join(f"- {txt}" for txt in search_context(question))
    prompt = {
        "role": "system",
        "content": f"Contexte :\n{context}\n\nUtilise ce contexte pour répondre."
    }

    # Update local history
    chat_history.append({"role": "user", "content": question})

    def stream():
        response = chat(
            model="llama3",
            messages=chat_history + [prompt, {"role": "user", "content": question}],
            stream=True
        )
        collected = ""
        for chunk in response:
            token = chunk["message"]["content"]
            collected += token
            yield token
        chat_history.append({"role": "assistant", "content": collected})

    return Response(stream(), content_type="text/plain")

@app.route("/")
def serve_ui():
    return send_from_directory(".", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(".", path)

if __name__ == "__main__":
    app.run(debug=True)
