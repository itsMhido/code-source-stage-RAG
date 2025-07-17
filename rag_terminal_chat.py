import faiss
import pickle
from sentence_transformers import SentenceTransformer
from ollama import chat

# Load FAISS index
index = faiss.read_index("fos_index.faiss")
with open("fos_metadata.pkl", "rb") as f:
    metas = pickle.load(f)

# Load embedding model for similarity search
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize conversation history
chat_history = [
    {
        "role": "system",
        "content": (
            "Tu es un assistant intelligent qui répond à des questions d'utilisateurs "
            "en se basant uniquement sur les documents fournis (questions/réponses d’archives internes). "
            "Si les informations ne sont pas présentes, dis que tu ne peux pas répondre."
        )
    }
]

# Search relevant context from FAISS
def search_context(query, k=5):
    q_vec = embedder.encode([query]).astype("float32")
    scores, idxs = index.search(q_vec, k)
    return [metas[i]['text'] for i in idxs[0] if i < len(metas)]

# Function to send question + retrieved context + history to Ollama
def generate_response(question, model_name="llama3"):
    retrieved = search_context(question)
    context_text = "\n".join(f"- {c}" for c in retrieved)

    # Add a context-aware system message just for this question
    context_message = {
        "role": "system",
        "content": f"Contexte:\n{context_text}\n\nUtilise ce contexte pour répondre à la question suivante."
    }

    # Build temporary messages for this turn
    messages = chat_history + [context_message, {"role": "user", "content": question}]

    response = chat(model=model_name, messages=messages)
    answer = response["message"]["content"]

    # Save user input and model response in history
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    return answer

# 💬 Terminal chat loop
def main():
    print("🧠 RAG Chatbot (Llama 3 via Ollama). Tape 'exit' pour quitter.\n")

    while True:
        user_input = input("👤 Vous : ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Fin de la session.")
            break

        try:
            reply = generate_response(user_input)
            print(f"🤖 Bot : {reply}\n")
        except Exception as e:
            print(f"⚠️ Erreur : {e}")

if __name__ == "__main__":
    main()
