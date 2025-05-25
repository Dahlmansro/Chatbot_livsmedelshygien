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
    /* Bakgrundsfärg för hela appen */
    .stApp {
        background-color: rgb(37, 150, 190);
        height: 100vh;
    }

    /* Dölj Streamlit-meny och footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}

    /* Chat-container styling */
    .chat-container {
        height: 70vh;
        overflow-y: auto;
        padding: 20px;
        margin-bottom: 20px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }

    /* Användarmeddelande styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0 10px auto;
        max-width: 80%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        font-size: 1.1rem;
        word-wrap: break-word;
    }

    /* Bot-svar styling */
    .bot-message {
        background: rgba(255, 255, 255, 0.95);
        color: #333;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px auto 10px 0;
        max-width: 80%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        font-size: 1.1rem;
        line-height: 1.5;
        word-wrap: break-word;
    }

    .bot-message .bot-icon {
        color: rgb(37, 150, 190);
        font-weight: bold;
        margin-bottom: 8px;
        display: block;
    }

    /* Detaljerad knapp styling */
    .detailed-button-container {
        text-align: left;
        margin-top: 10px;
    }

    .detailed-button {
        background: linear-gradient(135deg, rgb(37, 150, 190) 0%, #1e7e9a 100%);
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 15px;
        font-size: 0.9rem;
        cursor: pointer;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }

    .detailed-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* Input-område längst ner */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(37, 150, 190, 0.95);
        padding: 20px;
        backdrop-filter: blur(10px);
        border-top: 2px solid rgba(255,255,255,0.3);
        z-index: 1000;
    }

    /* Input-fält styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.95);
        border: 2px solid white;
        border-radius: 25px;
        padding: 15px 20px;
        font-size: 1.1rem;
        width: 100%;
    }

    .stTextInput > div > div > input::placeholder {
        color: #666 !important;
        opacity: 1 !important;
    }

    /* Skicka-knapp styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }

    /* Styling för huvudtitel */
    .main-title {
        color: white;
        text-align: center;
        font-size: 2.2rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    /* Välkomstmeddelande */
    .welcome-message {
        background: rgba(255, 255, 255, 0.95);
        color: #333;
        padding: 20px;
        border-radius: 20px;
        margin: 20px auto;
        max-width: 80%;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .welcome-message .bot-icon {
        color: rgb(37, 150, 190);
        font-size: 1.5rem;
        margin-bottom: 10px;
    }

    /* Spinner styling */
    .stSpinner {
        color: white;
    }

    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }

    .chat-container::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 3px;
    }

    .chat-container::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.3);
        border-radius: 3px;
    }

    .chat-container::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.5);
    }

    /* Responsiv design */
    @media (max-width: 768px) {
        .user-message, .bot-message {
            max-width: 95%;
        }
        
        .main-title {
            font-size: 1.8rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# === FUNKTIONER ===

def cosine_similarity(vec1, vec2):
    """Beräknar cosine similarity mellan två vektorer"""
    try:
        dot_product = dot(vec1, vec2)
        norm1 = norm(vec1)
        norm2 = norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    except Exception as e:
        logging.error(f"Fel i cosine_similarity: {e}")
        return 0.0

def create_embeddings(text, model="models/embedding-001", task_type="semantic_similarity"):
    """Skapar embeddings för given text"""
    try:
        return genai.embed_content(
            model=model,
            content=text,
            task_type=task_type
        )
    except Exception as e:
        logging.error(f"Fel vid skapande av embeddings: {e}")
        return None

def semantic_search(query, chunks, embeddings, k=5):
    """Utför semantic search och returnerar de mest relevanta chunks"""
    try:
        query_result = create_embeddings(query)
        if not query_result:
            logging.warning("Kunde inte skapa embedding för frågan")
            return chunks[:k]  # Fallback
            
        query_embedding = query_result["embedding"]
        similarity_scores = []

        for i, chunk_embedding in enumerate(embeddings):
            try:
                similarity = cosine_similarity(query_embedding, chunk_embedding)
                similarity_scores.append((i, similarity))
            except Exception as e:
                logging.warning(f"Fel vid beräkning av similarity för chunk {i}: {e}")
                similarity_scores.append((i, 0))

        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [index for index, _ in similarity_scores[:k]]
        return [chunks[i] for i in top_indices]
    
    except Exception as e:
        logging.error(f"Fel i semantic_search: {e}")
        return chunks[:k]  # Fallback

def generate_response(query, context, model, detailed=False):
    """Genererar svar baserat på fråga och kontext"""
    if detailed:
        system_prompt = """
        Ge ett detaljerat svar baserat på kontexten. Svara i hela meningar och avsluta med en naturlig avslutning.
        Skriv enkelt och tydligt på svenska. Använd flera stycken om det passar.
        Avsluta alltid svaret med en fullständig mening – inte mitt i.
        """
    else:
        system_prompt = """
        Svara alltid med max 2 hela meningar. Avsluta alltid svaret med en punkt. 
        Om du når tokengränsen, avsluta med en fullständig mening, även om det innebär att du måste korta svaret något. 
        Använd enkel svenska utan listor eller markdown. 
        Lägg till frågan: 'Vill du ha ett mer detaljerat svar?' sist.
        """
    
    user_prompt = f"Frågan är: {query}\n\nHär är kontexten:\n{context}"
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    try:
        max_tokens = 500 if detailed else 100
        response = model.generate_content(
            full_prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "stop_sequences": ["\n\n", "\nFråga:"]
            }
        )

        return response.text.strip()
    except Exception as e:
        logging.error(f"Fel vid generering av svar: {e}")
        return "Jag kan inte svara på din fråga just nu. Försök igen senare."

def load_data():
    """Laddar chunks och embeddings från pickle-filer"""
    try:
        # Kontrollera att filerna existerar
        if not os.path.exists("semantic_chunks.pkl"):
            st.error("Filen semantic_chunks.pkl hittades inte!")
            st.stop()
            
        if not os.path.exists("embeddings_chunks.pkl"):
            st.error("Filen embeddings_chunks.pkl hittades inte!")
            st.stop()

        with open("semantic_chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        with open("embeddings_chunks.pkl", "rb") as f:
            embeddings = pickle.load(f)
            
        return chunks, embeddings
    
    except Exception as e:
        st.error(f"Fel vid laddning av data: {e}")
        st.stop()

def get_api_key():
    """Hämtar API-nyckel från antingen .env-fil eller Streamlit secrets"""
    # Först försök ladda från .env (för lokal utveckling)
    load_dotenv()
    
    # Försök hämta från olika källor
    api_key = (
        os.getenv("GEMINI_API_KEY") or 
        os.getenv("API_KEY") or
        st.secrets.get("GEMINI_API_KEY", None) or
        st.secrets.get("API_KEY", None)
    )
    
    return api_key

def display_chat_message(message_type, content, show_detailed_button=False, message_id=None):
    """Visar ett chat-meddelande med korrekt styling"""
    if message_type == "user":
        st.markdown(f'''
            <div class="user-message">
                {content}
            </div>
        ''', unsafe_allow_html=True)
    else:  # bot message
        button_html = ""
        if show_detailed_button and message_id:
            button_html = f'''
                <div class="detailed-button-container">
                    <button class="detailed-button" onclick="window.parent.postMessage({{type: 'detailed_request', messageId: '{message_id}'}}, '*')">
                        📋 Mer detaljerat svar
                    </button>
                </div>
            '''
        
        st.markdown(f'''
            <div class="bot-message">
                <span class="bot-icon">🍽️ Chatbot</span>
                {content}
                {button_html}
            </div>
        ''', unsafe_allow_html=True)

def initialize_chat_history():
    """Initialiserar chat-historiken med välkomstmeddelande"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.message_counter = 0
        
        # Lägg till välkomstmeddelande
        welcome_msg = """
        Hej! Jag är din assistent för livsmedelshygien. 🍽️
        
        Jag kan hjälpa dig med frågor om HACCP, temperaturkrav, rengöring, allergeninformation och mycket mer baserat på Visitas branschriktlinjer.
        
        **Testa gärna att fråga:** "Vad är HACCP?"
        """
        
        st.session_state.chat_history.append({
            'type': 'bot',
            'content': welcome_msg,
            'id': 'welcome',
            'show_detailed': False
        })

# === HUVUDAPPLIKATION ===

def main():
    # Konfiguration för bred layout
    st.set_page_config(
        page_title="Livsmedelshygien Chatbot",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Ladda CSS
    load_css()
    
    # Hämta API-nyckel
    API_KEY = get_api_key()
    
    if not API_KEY:
        st.error("""
        ❌ API-nyckel saknas! 
        
        **För lokal utveckling:** Skapa en .env-fil med:
        ```
        GEMINI_API_KEY=din_api_nyckel_här
        ```
        
        **För Streamlit Cloud:** Lägg till API-nyckeln i Secrets-sektionen.
        """)
        st.stop()
    
    # Konfigurera Gemini
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("models/gemini-2.0-flash")
    except Exception as e:
        st.error(f"Fel vid konfiguration av Gemini: {e}")
        st.stop()

    # Ladda data
    chunks, embeddings = load_data()
    
    # Initialisera chat-historik
    initialize_chat_history()

    # Titel
    st.markdown('<h1 class="main-title">🍽️ Livsmedelshygien Chatbot</h1>', unsafe_allow_html=True)

    # Chat-container för meddelanden
    chat_container = st.container()
    
    # Skapa huvudlayout
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col2:
        # Visa chat-historik
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            for message in st.session_state.chat_history:
                display_chat_message(
                    message['type'], 
                    message['content'],
                    message.get('show_detailed', False),
                    message.get('id')
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Mellanrum för input-området
        st.markdown('<div style="height: 120px;"></div>', unsafe_allow_html=True)
        
        # Fast input-område längst ner
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        # Input-fält och knapp
        input_col, button_col = st.columns([6, 1])
        
        with input_col:
            query = st.text_input(
                "", 
                placeholder="Skriv din fråga om livsmedelshygien här... (t.ex. 'Vad är HACCP?')",
                key="chat_input",
                label_visibility="collapsed"
            )
        
        with button_col:
            send_button = st.button("➤", help="Skicka meddelande", key="send_btn")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Hantera nya meddelanden
    if query.strip() and (send_button or query != st.session_state.get('last_processed_query', '')):
        st.session_state.last_processed_query = query
        
        # Lägg till användarmeddelande
        st.session_state.chat_history.append({
            'type': 'user',
            'content': query,
            'id': f"user_{st.session_state.message_counter}"
        })
        st.session_state.message_counter += 1
        
        # Generera bot-svar
        with st.spinner("🔍 Tänker..."):
            try:
                # Sök relevanta chunks
                normalized_query = query.lower().strip()
                relevant_chunks = semantic_search(normalized_query, chunks, embeddings, k=5)
                context = "\n\n".join(relevant_chunks)
                
                # Generera svar
                answer = generate_response(normalized_query, context, model)
                
                # Kontrollera om vi ska visa detaljerad knapp
                show_detailed = "Vill du ha ett mer detaljerat svar?" in answer
                
                # Lägg till bot-svar
                message_id = f"bot_{st.session_state.message_counter}"
                st.session_state.chat_history.append({
                    'type': 'bot',
                    'content': answer,
                    'id': message_id,
                    'show_detailed': show_detailed,
                    'query': query,
                    'context': context
                })
                st.session_state.message_counter += 1
                
                # Rensa input-fältet
                st.session_state.chat_input = ""
                
                # Rerun för att visa nya meddelanden
                st.rerun()
                
            except Exception as e:
                st.error(f"Ett fel inträffade: {e}")
                logging.error(f"Fel vid generering av svar: {e}")

    # Hantera detaljerade svar-förfrågningar
    if st.button("📋 Ge detaljerat svar", key="detailed_btn", help="Klicka för att få mer information"):
        # Hitta senaste bot-meddelandet som kan få detaljerat svar
        for i in range(len(st.session_state.chat_history) - 1, -1, -1):
            message = st.session_state.chat_history[i]
            if message['type'] == 'bot' and message.get('show_detailed', False):
                with st.spinner("📝 Skapar detaljerat svar..."):
                    try:
                        detailed_answer = generate_response(
                            message['query'].lower().strip(),
                            message['context'],
                            model,
                            detailed=True
                        )
                        
                        # Lägg till detaljerat svar
                        st.session_state.chat_history.append({
                            'type': 'bot',
                            'content': f"**Detaljerat svar:**\n\n{detailed_answer}",
                            'id': f"detailed_{st.session_state.message_counter}",
                            'show_detailed': False
                        })
                        st.session_state.message_counter += 1
                        
                        # Markera att detaljerat svar har getts
                        message['show_detailed'] = False
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Ett fel inträffade: {e}")
                        logging.error(f"Fel vid generering av detaljerat svar: {e}")
                break

    # JavaScript för smooth scrolling
    st.markdown("""
    <script>
    function scrollToBottom() {
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }
    
    // Scroll när sidan laddas
    setTimeout(scrollToBottom, 100);
    
    // Scroll efter nya meddelanden
    const observer = new MutationObserver(scrollToBottom);
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        observer.observe(chatContainer, { childList: true, subtree: true });
    }
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()