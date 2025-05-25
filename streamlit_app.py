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
    }

    /* Styling för huvudtitel */
    .main-title {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    /* Chat meddelande styling */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    .user-message {
        background-color: rgba(102, 126, 234, 0.8);
        color: white;
        margin-left: 2rem;
    }

    .bot-message {
        background-color: rgba(255, 255, 255, 0.95);
        color: #333;
        margin-right: 2rem;
    }

    .welcome-message {
        background-color: rgba(255, 255, 255, 0.95);
        color: #333;
        border-left: 4px solid rgb(37, 150, 190);
        margin: 1rem 0;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.95);
        border: 2px solid white;
        border-radius: 10px;
        padding: 10px;
        font-size: 1.1rem;
    }

    .stTextInput > div > div > input::placeholder {
        color: #666 !important;
        opacity: 1 !important;
    }

    /* Button styling */
    .stButton > button {
        background-color: white;
        color: rgb(37, 150, 190);
        border: 2px solid white;
        border-radius: 8px;
        font-weight: bold;
        padding: 8px 16px;
        font-size: 1rem;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: rgb(37, 150, 190);
        color: white;
        border: 2px solid white;
    }

    /* Send button styling */
    .send-button > button {
        background-color: rgb(37, 150, 190);
        color: white;
        border: 2px solid rgb(37, 150, 190);
        border-radius: 50px;
        font-weight: bold;
        padding: 8px 16px;
        font-size: 1.1rem;
    }

    .send-button > button:hover {
        background-color: white;
        color: rgb(37, 150, 190);
        border: 2px solid rgb(37, 150, 190);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 5px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Spacing */
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    /* Input section styling */
    .input-section {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin-top: 2rem;
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

def initialize_chat():
    """Initialiserar chat-sessionen"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.show_welcome = True

def display_welcome_message():
    """Visar välkomstmeddelande"""
    st.markdown("""
    <div class="chat-message welcome-message">
        <h3>🍽️ Välkommen till Livsmedelshygien Chatbot!</h3>
        <p>Jag kan hjälpa dig med frågor om:</p>
        <ul>
            <li>HACCP-systemet</li>
            <li>Temperaturkrav och förvaring</li>
            <li>Rengöring och desinfektion</li>
            <li>Allergeninformation</li>
            <li>Branschriktlinjer från Visita</li>
        </ul>
        <p><strong>Testa gärna att fråga:</strong> "Vad är HACCP?"</p>
    </div>
    """, unsafe_allow_html=True)

def display_chat_history():
    """Visar chat-historiken"""
    if st.session_state.show_welcome:
        display_welcome_message()
    
    for i, message in enumerate(st.session_state.chat_history):
        if message['type'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>Du:</strong> {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🍽️ Chatbot:</strong> {message['content']}
            </div>
            """, unsafe_allow_html=True)
            
            # Visa detaljerad knapp om tillgänglig
            if message.get('show_detailed', False):
                if st.button(f"📋 Ge mer detaljerat svar", key=f"detailed_{i}"):
                    with st.spinner("📝 Skapar detaljerat svar..."):
                        try:
                            detailed_answer = generate_response(
                                message['query'].lower().strip(),
                                message['context'],
                                st.session_state.model,
                                detailed=True
                            )
                            
                            # Lägg till detaljerat svar i historiken
                            st.session_state.chat_history.append({
                                'type': 'bot',
                                'content': f"**Detaljerat svar:**\n\n{detailed_answer}",
                                'show_detailed': False
                            })
                            
                            # Markera att detaljerat svar har getts
                            message['show_detailed'] = False
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Ett fel inträffade: {e}")
                            logging.error(f"Fel vid generering av detaljerat svar: {e}")

# === HUVUDAPPLIKATION ===

def main():
    # Sidkonfiguration
    st.set_page_config(
        page_title="Livsmedelshygien Chatbot",
        page_icon="🍽️",
        layout="wide"
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
        st.session_state.model = genai.GenerativeModel("models/gemini-2.0-flash")
    except Exception as e:
        st.error(f"Fel vid konfiguration av Gemini: {e}")
        st.stop()

    # Ladda data
    chunks, embeddings = load_data()
    
    # Initialisera chat
    initialize_chat()

    # Layout
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col2:
        # Titel
        st.markdown('<h1 class="main-title">🍽️ Livsmedelshygien Chatbot</h1>', unsafe_allow_html=True)
        
        # Chat-område
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        display_chat_history()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Input-sektion längst ner
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### 💬 Ställ din fråga:")
        
        # Input-fält och knapp
        input_col, button_col = st.columns([4, 1])
        
        with input_col:
            query = st.text_input(
                "Fråga:",
                placeholder="T.ex. Vad är HACCP? Vilka temperaturkrav gäller för kött?",
                key="user_input",
                label_visibility="collapsed"
            )
        
        with button_col:
            st.markdown('<div class="send-button">', unsafe_allow_html=True)
            send_button = st.button("➤ Skicka", key="send_btn")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Hantera ny fråga
    if query.strip() and (send_button or query != st.session_state.get('last_query', '')):
        st.session_state.last_query = query
        st.session_state.show_welcome = False  # Dölj välkomstmeddelandet
        
        # Lägg till användarfråga
        st.session_state.chat_history.append({
            'type': 'user',
            'content': query
        })
        
        # Generera svar
        with st.spinner("🔍 Söker efter relevant information..."):
            try:
                # Sök relevanta chunks
                normalized_query = query.lower().strip()
                relevant_chunks = semantic_search(normalized_query, chunks, embeddings, k=5)
                context = "\n\n".join(relevant_chunks)
                
                # Generera svar
                answer = generate_response(normalized_query, context, st.session_state.model)
                
                # Kontrollera om vi ska visa detaljerad knapp
                show_detailed = "Vill du ha ett mer detaljerat svar?" in answer
                
                # Lägg till bot-svar
                st.session_state.chat_history.append({
                    'type': 'bot',
                    'content': answer,
                    'show_detailed': show_detailed,
                    'query': query,
                    'context': context
                })
                
                # Rensa input
                st.session_state.user_input = ""
                
                # Uppdatera sidan
                st.rerun()
                
            except Exception as e:
                st.error(f"Ett fel inträffade: {e}")
                logging.error(f"Fel vid generering av svar: {e}")

    # Info-sektion
    with st.expander("ℹ️ Om denna chatbot"):
        st.markdown("""
        **Livsmedelshygien Chatbot**
        
        - Denna chatbot svarar på frågor om livsmedelshygien baserat på Visitas branschriktlinjer
        - Källa: [Visitas Branschriktlinjer](https://visita.se/app/uploads/2021/06/Visita_Branschriktlinjer-print_2021.pdf)
        
        **Så här använder du chatten:**
        1. Skriv din fråga i textfältet längst ner
        2. Tryck Enter eller klicka på "➤ Skicka"
        3. Konversationen visas ovanför och växer uppåt
        4. Klicka på "📋 Ge mer detaljerat svar" för utförligare information
        
        **Exempel på frågor:**
        - Vad är HACCP?
        - Vilka temperaturkrav gäller för kött?
        - Hur ofta ska kylskåp rengöras?
        - Vad är kritiska kontrollpunkter?
        """)

if __name__ == "__main__":
    main()