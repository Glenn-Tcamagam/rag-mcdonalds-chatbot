import os
import boto3
import json
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ------------------------------------------------------------
# 0. RÉCUPÉRER LA CLÉ OPENAI DEPUIS AWS SECRETS MANAGER
# ------------------------------------------------------------
def get_openai_key_from_aws():
    client = boto3.client("secretsmanager", region_name="eu-north-1")

    response = client.get_secret_value(SecretId="my_rag_secrets")
    secret = json.loads(response["SecretString"])

    return secret["OPENAI_API_KEY"]


# ------------------------------------------------------------
# 1. CHARGER UN PDF
# ------------------------------------------------------------
def load_pdf(path_pdf):
    loader = PyPDFLoader(path_pdf)
    return loader.load()


# ------------------------------------------------------------
# 2. SPLIT DES DOCUMENTS
# ------------------------------------------------------------
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )
    return splitter.split_documents(docs)


# ------------------------------------------------------------
# 3. VECTORSTORE + EMBEDDINGS OPENAI
# ------------------------------------------------------------
def create_vectorstore(splits):
    api_key = get_openai_key_from_aws()

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=api_key
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vectorstore.as_retriever(search_kwargs={"k": 7})


# -------------------------------------------------------------------
# 4. MÉMOIRE DYNAMODB (PRO)
# -------------------------------------------------------------------
class DynamoDBMemory:
    def __init__(self, table_name="rag_memory", session_id="default"):
        self.session_id = session_id
        self.table = boto3.resource("dynamodb", region_name="eu-north-1").Table(table_name)

    def save_message(self, role, content):
        self.table.put_item(
            Item={
                "session_id": self.session_id,
                "timestamp": int(time.time() * 1000),
                "role": role,
                "content": content
            }
        )

    def load_messages(self):
        resp = self.table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("session_id").eq(self.session_id),
            ScanIndexForward=True
        )

        messages = resp.get("Items", [])

        history = ""
        for msg in messages:
            prefix = "Utilisateur: " if msg["role"] == "user" else "Assistant: "
            history += prefix + msg["content"] + "\n\n"

        return history.strip()


# ------------------------------------------------------------
# 5. RAG CHAIN AVEC MÉMOIRE
# ------------------------------------------------------------
def create_rag(retriever, session_id="default"):

    # Mémoire DynamoDB
    memory = DynamoDBMemory(session_id=session_id)

    api_key = get_openai_key_from_aws()
    os.environ["OPENAI_API_KEY"] = api_key

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    # Ajout de {memory} dans le prompt
    system_prompt = (
    "Tu es un assistant expert spécialisé EXCLUSIVEMENT dans les burgers de McDonald’s.\n"
    "Tu réponds aux questions des utilisateurs en t’appuyant sur un système RAG professionnel.\n\n"

    "📌 SOURCES AUTORISÉES (par ordre de priorité) :\n"
    "1️⃣ La mémoire de la conversation (questions et réponses précédentes)\n"
    "2️⃣ Les documents PDF McDonald’s indexés (burgers, ingrédients, allergènes, valeurs nutritionnelles, prix)\n"
    "3️⃣ Ton raisonnement logique UNIQUEMENT pour expliquer ou reformuler les informations trouvées\n\n"

    "🧠 RÈGLES IMPORTANTES :\n"
    "- Tu réponds UNIQUEMENT à propos des burgers McDonald’s.\n"
    "- Si une information est présente dans les documents, tu dois la donner clairement et précisément.\n"
    "- Tu ne dois JAMAIS inventer d’informations.\n"
    "- Si l’information n’est pas présente dans les documents, dis-le explicitement.\n"
    "- Si la question concerne un autre produit que les burgers (boissons, desserts, menus enfants, etc.), précise que ce n’est pas couvert.\n\n"

    "📄 MÉMOIRE DE LA CONVERSATION :\n{memory}\n\n"

    "📄 EXTRAITS DES DOCUMENTS (RAG) :\n{context}\n\n"

    "✍️ STYLE DE RÉPONSE ATTENDU :\n"
    "- Réponse claire et structurée\n"
    "- Listes à puces si nécessaire\n"
    "- Ton professionnel et pédagogique\n"
    "- Réponses complètes mais concises\n"
)


    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # Chaîne RAG + mémoire
    def rag_with_memory(user_input):
        conversation_memory = memory.load_messages()
        docs = retriever.invoke(user_input)
        context = format_docs(docs)

        inputs = {
            "memory": conversation_memory,
            "context": context,
            "input": user_input
        }

        answer = (prompt | llm | StrOutputParser()).invoke(inputs)

        # Sauvegarder en mémoire
        memory.save_message("user", user_input)
        memory.save_message("assistant", answer)

        return answer

    return rag_with_memory



# -----------------------------
# Helper: construire la chaîne RAG complète (utilisée par app.py)
# -----------------------------
import os  # si déjà importé dans le fichier, ça ne pose pas de problème

def get_rag_chain(session_id: str = "client_1", pdf_folder: str = "pdfs"):
    """
    Convenience function used by app.py.
    - Charge automatiquement tous les PDFs du dossier `pdf_folder`
    - Split/embeddings/vectorstore
    - Crée et retourne la fonction RAG (callable) prête à être utilisée.
    """
    # 1) récupérer fichiers pdf
    if not os.path.isdir(pdf_folder):
        raise ValueError(f"Dossier PDF introuvable: {pdf_folder}")

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise ValueError(f"Aucun fichier PDF trouvé dans {pdf_folder}.")

    # 2) charger tous les docs
    all_docs = []
    for pdf in pdf_files:
        path = os.path.join(pdf_folder, pdf)
        docs = load_pdf(path)
        all_docs.extend(docs)

    # 3) split + vectorstore
    splits = split_documents(all_docs)
    retriever = create_vectorstore(splits)

    # 4) créer la chaîne RAG avec mémoire (DynamoDB) — create_rag doit accepter session_id
    #    create_rag retourne un callable (par ex. rag_with_memory)
    return create_rag(retriever, session_id=session_id)
