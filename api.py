from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 👉 On importe ton moteur RAG
from rag_engine import get_rag_chain

# Création de l'app FastAPI
app = FastAPI(
    title="RAG McDonald's API",
    description="API RAG pour répondre aux questions sur les burgers McDonald's",
    version="1.0"
)

# ---------------------------
# CORS CONFIGURATION
# ---------------------------

origins = [
    "http://localhost:5501",   # portfolio en local (Live Server VSCode)
    "http://127.0.0.1:5501",
    "https://portfolio-tchamagamglenn.netlify.app"  # portfolio en prod
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],        # GET, POST, etc.
    allow_headers=["*"],
)

# 👉 Initialisation du RAG (chargé une seule fois au démarrage)
rag_chain = get_rag_chain()

# ---------
# Modèle de requête (ce que le client envoie)
# ---------
class ChatRequest(BaseModel):
    question: str

# ---------
# Modèle de réponse (ce que l'API renvoie)
# ---------
class ChatResponse(BaseModel):
    answer: str

# ---------
# Endpoint de test
# ---------
@app.get("/")
def health_check():
    return {"status": "API RAG opérationnelle"}

# ---------
# Endpoint principal RAG
# ---------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Reçoit une question en entrée
    Appelle le RAG
    Retourne la réponse
    """

    # Appel du RAG
    answer = rag_chain(request.question)

    return {
        "answer": answer
    }
