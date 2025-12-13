import streamlit as st
from rag_engine import get_rag_chain

# ------------------------------------------------------------
# CONFIG PAGE
# ------------------------------------------------------------
st.set_page_config(
    page_title="McDonald's Burger Assistant",
    page_icon="🍔",
    layout="wide"
)

# ------------------------------------------------------------
# STYLE GLOBAL (SOFT / PREMIUM)
# ------------------------------------------------------------
st.markdown("""
<style>

/* --- GLOBAL --- */
html, body, [class*="css"] {
    font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #fafafa;
}

/* --- HEADER --- */
.header {
    text-align: center;
    padding: 15px 0 5px 0;
}

.header-title {
    font-size: 30px;
    font-weight: 700;
    color: #1f1f1f;
}

.header-subtitle {
    font-size: 15px;
    color: #6b6b6b;
    margin-top: 5px;
}

/* --- CHAT CONTAINER --- */
.chat-container {
    max-height: 620px;
    overflow-y: auto;
    padding: 20px;
    margin-top: 15px;
    border-radius: 18px;
    background-color: #ffffff;
    border: 1px solid #eeeeee;
}

/* --- USER MESSAGE --- */
.user-bubble {
    background-color: #fff1d6;
    color: #1f1f1f;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 70%;
    float: right;
    clear: both;
    font-size: 15px;
}

/* --- BOT MESSAGE --- */
.bot-bubble {
    background-color: #f4f5f7;
    color: #1f1f1f;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 70%;
    float: left;
    clear: both;
    font-size: 15px;
}

/* --- FOOTER HINT --- */
.footer-hint {
    text-align: center;
    font-size: 13px;
    color: #8a8a8a;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# INITIALISATION RAG + MÉMOIRE
# ------------------------------------------------------------
if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = get_rag_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("""
<div class="header">
    <div class="header-title">🍔 Assistant Burgers McDonald’s</div>
    <div class="header-subtitle">
        Ingrédients • Allergènes • Calories • Prix • Nutri-Score
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# CHAT HISTORY
# ------------------------------------------------------------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(
            f"<div class='user-bubble'>{message['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='bot-bubble'>{message['content']}</div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# USER INPUT
# ------------------------------------------------------------
user_input = st.chat_input(
    "Pose une question sur les burgers McDonald’s…"
)

if user_input:
    # Ajouter message utilisateur
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    # Appel RAG
    answer = st.session_state["rag_chain"](user_input)

    # Ajouter message assistant
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

    # Scroll automatique
    st.rerun()

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown(
    "<div class='footer-hint'>Données issues de la base de connaissance de Glenn Tchamagam • \nJe suis Glenn Tchamagam Ingénieur en génie lociel et actuellement etudiant en MSC1 Data management à ECE Paris. Je suis à la recherche d'un stage académique à partir de Avril 2026 et une alternance à partir de Septembre 2026</div>",
    unsafe_allow_html=True
)
