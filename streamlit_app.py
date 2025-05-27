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

# === CUSTOM CSS ===
def load_css():
    st.markdown("""
    <style>
    ...
    </style>
    """, unsafe_allow_html=True)  # <-- CSS-koden förkortad här för läsbarhet

# === FUNKTIONER ===
# (samma som tidigare: cosine_similarity, create_embeddings, semantic_search, generate_response, load_data, get_api_key, initialize_chat, display_welcome_message, display_chat_history)

# === HUVUDAPPLIKATION ===
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
        
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### 💬 Ställ din fråga:")

        with st.form("chat_form", clear_on_submit=True):
            query = st.text_input("Fråga:", placeholder="Skriv din fråga här...", label_visibility="collapsed", key="user_query")
            st.markdown('<div class="send-button" style="margin-top: 0.5rem;">', unsafe_allow_html=True)
            send_button = st.form_submit_button("➤ Skicka")
            st.markdown('</div>', unsafe_allow_html=True)

        # 🔥 NY KOD FÖR ATT HANTERA KNAPPTRYCKNING 🔥
        if send_button and query.strip():
            with st.spinner("🔍 Letar efter svar..."):
                try:
                    relevant_chunks = semantic_search(query, chunks, embeddings)
                    context = "\n\n".join(relevant_chunks)
                    answer = generate_response(query, context, st.session_state.model)

                    st.session_state.chat_history.append({'type': 'user', 'content': query})
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'content': answer,
                        'show_detailed': True,
                        'query': query,
                        'context': context
                    })

                    st.session_state.show_welcome = False
                    st.rerun()

                except Exception as e:
                    st.error("Ett fel uppstod vid generering av svaret.")
                    logging.error(f"Fel vid användarfråga: {e}")

        with st.expander("ℹ️ Om denna chatbot"):
            st.markdown("""
            **Livsmedelshygien Chatbot**
            - Denna chatbot svarar på frågor om livsmedelshygien baserat på Visitas branschriktlinjer  
            - Källa: [Visitas Branschriktlinjer](https://visita.se/app/uploads/2021/06/Visita_Branschriktlinjer-print_2021.pdf)
            """)

if __name__ == "__main__":
    main()
