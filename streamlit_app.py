import streamlit as st
import pickle
import os
import google.generativeai as genai
from dotenv import load_dotenv
from numpy import dot
from numpy.linalg import norm
import logging

# === KONFIGURATION AV LOGGING ===
logging.basicConfig(level=logging.INFO)

# === LADDA CSS ===
def load_css():
    st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 20px;
    }
    .chat-container {
        background-color: #f8f8f8;
        padding: 20px;
        border-radius: 10px;
        height: auto;
        overflow-y: auto;
        margin-bottom: 20px;
    }
    .input-section {
        margin-top: 20px;
    }
    .example-buttons button {
        margin-right: 0.5em;
        margin-bottom: 0.5em;
    }
    </style>
    """, unsafe_allow_html=True)

# === FUNKTIONER ===
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def create_embeddings(text):
    return genai.embed_content(content=text, model="models/embedding-001", task_type="semantic_similarity")

def semantic_search(query, chunks, embeddings, k=5):
    query_embedding = create_embeddings(query)["embedding"]
    similarities = [(i, cosine_similarity(query_embedding, emb)) for i, emb in enumerate(embeddings)]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [chunks[i] for i, _ in similarities[:k]]

def generate_response(query, context, model):
    prompt = f"Fråga: {query}\n\nKälltext:\n{context}\n\nSvar:"
    return model.generate_content(prompt).text

def get_api_key():
    load_dotenv()
    return os.getenv("GOOGLE_API_KEY")

def load_data():
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    return chunks, embeddings

def initialize_chat():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_welcome" not in st.session_state:
        st.session_state.show_welcome = True
    if "query_to_send" not in st.session_state:
        st.session_state.query_to_send = ""

def display_chat_history():
    for message in st.session_state.chat_history:
        if message['type'] == 'user':
            st.markdown(f"🧍 **Du:** {message['content']}")
        elif message['type'] == 'bot':
            st.markdown(f"🤖 **Chatbot:** {message['content']}")

def handle_query(query, chunks, embeddings, model):
    relevant_chunks = semantic_search(query, chunks, embeddings)
    context = "\n\n".join(relevant_chunks)
    answer = generate_response(query, context, model)

    st.session_state.chat_history.append({'type': 'user', 'content': query})
    st.session_state.chat_history.append({
        'type': 'bot',
        'content': answer,
        'show_detailed': True,
        'query': query,
        'context': context
    })
    st.session_state.show_welcome = False
    st.session_state.query_to_send = ""  # nollställ

# === HUVUDPROGRAM ===
def main():
    st.set_page_config(page_title="Livsmedelshygien Chatbot", page_icon="🍽️", layout="wide")
    load_css()

    API_KEY = get_api_key()
    if not API_KEY:
        st.error("❌ API-nyckel saknas!")
        st.stop()

    try:
        genai.configure(api_key=API_KEY)
        st.session_state.model = genai.GenerativeModel("models/gemini-2.0-flash")
    except Exception as e:
        st.error(f"Fel vid konfiguration av Gemini: {e}")
        st.stop()

    chunks, embeddings = load_data()
    initialize_chat()

    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.markdown('<h1 class="main-title">🍽️ Livsmedelshygien Chatbot</h1>', unsafe_allow_html=True)
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        display_chat_history()
        st.markdown('</div>', unsafe_allow_html=True)

        # 🔘 Klickbara exempelfrågor
        st.markdown("#### 💡 Exempelfrågor:")
        example_questions = [
            "Vad är HACCP?",
            "Hur ofta ska man byta förkläde i köket?",
            "Får man servera råbiff?",
        ]
        for example in example_questions:
            if st.button(example, key=f"btn_{example}"):
                st.session_state.query_to_send = example
                st.rerun()

        # 🔍 Om användaren har tryckt på en exempelfråga
        if st.session_state.query_to_send:
            with st.spinner("🔍 Letar efter svar..."):
                handle_query(st.session_state.query_to_send, chunks, embeddings, st.session_state.model)
                st.rerun()

        # 🧾 Användarformulär
        st.markdown("### 💬 Ställ din fråga:")
        with st.form("chat_form", clear_on_submit=True):
            query = st.text_input("Fråga", placeholder="Skriv din fråga här...", label_visibility="collapsed")
            submitted = st.form_submit_button("➤ Skicka")

        if submitted and query.strip():
            with st.spinner("🔍 Letar efter svar..."):
                handle_query(query, chunks, embeddings, st.session_state.model)
                st.rerun()

        with st.expander("ℹ️ Om denna chatbot"):
            st.markdown("""
            **Livsmedelshygien Chatbot**
            - Denna chatbot svarar på frågor om livsmedelshygien baserat på Visitas branschriktlinjer.  
            - Källa: [Visitas Branschriktlinjer](https://visita.se/app/uploads/2021/06/Visita_Branschriktlinjer-print_2021.pdf)
            """)

if __name__ == "__main__":
    main()