import streamlit as st
import pickle
import os
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np

st.set_page_config(
    page_title="🍽️ Livsmedelshygien Expert",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def load_advanced_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp { 
        font-family: 'Inter', sans-serif;
        background-color: rgb(37, 150, 190);
    }
                    
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        color: white;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        text-align: center;
        font-size: 4rem; 
        color: rgba(255,255,255,0.9);
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
    }
    
    .user-message {
        background: white;
        color: black;
        padding: 1rem 1.5rem;
        border-radius: 20px;
        margin: 1rem auto;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        position: relative;
        animation: slideInRight 0.3s ease-out;
        max-width: 80%;
        width: fit-content;
    }
        
    .bot-message {
        background: white;
        color: black;
        padding: 1rem 1.5rem;
        border-radius: 20px;
        margin: 1rem auto;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
        position: relative;
        animation: slideInLeft 0.3s ease-out;
        max-width: 80%;
        width: fit-content;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .welcome-message {
        text-align: center;
        padding: 2rem;
        color: black
        font-size: 1.1rem;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600;
    }
    
    .example-section {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .example-title {
        color: black;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .input-section {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .input-title {
        color: #667eea;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stTextInput > div > div > input {
        border-radius: 15px;
        border: 2px solid #e1e8ed;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.2);
    }
    
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
        font-weight: 500;
        color: #667eea;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 10px;
    }
    
    .stError {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        border-radius: 10px;
    }
    
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .chat-container {
            padding: 1rem;
        }
        .user-message, .bot-message {
            margin-left: 0.5rem;
            margin-right: 0.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def get_api_key():
    if os.path.exists(".env"):
        load_dotenv()
        api_key = os.getenv("API_KEY")
        if api_key:
            return api_key
    try:
        return st.secrets["API_KEY"]
    except:
        return None


def cosine_similarity(vec1, vec2):
    try:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    except:
        return 0.0


def create_embeddings(text):
    try:
        result = genai.embed_content(
            content=text,
            model="models/text-embedding-004",
            task_type="retrieval_query"
        )
        return result
    except Exception as e:
        st.error(f"Embedding fel: {e}")
        return None


def semantic_search(query, chunks, embeddings, k=5):
    try:
        query_result = create_embeddings(query)
        if not query_result:
            return chunks[:k]

        query_embedding = query_result['embedding']
        similarities = []

        for i, chunk_embedding in enumerate(embeddings):
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((i, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [chunks[i] for i, _ in similarities[:k]]

    except Exception as e:
        return chunks[:k]


def generate_response(query, chunks, embeddings, model, detailed=False):
    try:
        relevant_chunks = semantic_search(query, chunks, embeddings, k=5)
        context = "\n\n".join(relevant_chunks)

        if detailed:
            system_prompt = """
            Ge ett detaljerat och utförligt svar på svenska. Använd flera stycken och förklara grundligt. 
            Inkludera praktiska exempel och specifika detaljer från kontexten.
            Använd punktlistor och numrerade listor för att förtydliga viktiga punkter och steg.
            Strukturera svaret med rubriker och listor för att göra det lättläst.
            KRITISKT VIKTIGT: Du måste ALLTID avsluta med en fullständig mening och punkt. 
            Om du märker att du närmar dig tokengränsen, avsluta det aktuella stycket med 
            en naturlig slutsats istället för att påbörja ett nytt ämne. Skriv aldrig ofullständiga meningar.
            """
            max_tokens = 800
        else:
            system_prompt = """
            Svara alltid med max 2 hela meningar. Avsluta alltid svaret med en punkt. Om du når tokengränsen, avsluta med en fullständig mening, även om det innebär att du måste korta svaret något. 
            Använd enkel svenska utan listor eller markdown. Lägg INTE till frågan om detaljerat svar - det hanteras separat.
            """
            max_tokens = 100

        full_prompt = f"""{system_prompt}

        Fråga: {query}
        
        Här är kontexten:
        {context}"""

        response = model.generate_content(
            full_prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": 0.7,  # Lite mer kreativitet för bättre avslutningar
                "top_p": 0.8
            }
        )

        return response.text

    except Exception as e:
        return f"Tyvärr uppstod ett fel vid generering av svar. Försök igen."


@st.cache_data
def load_data():
    try:
        with open("semantic_chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        with open("embeddings_chunks.pkl", "rb") as f:
            embeddings = pickle.load(f)
        return chunks, embeddings
    except Exception as e:
        st.error(f"Fel vid laddning av data: {e}")
        st.stop()


def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "query_to_send" not in st.session_state:
        st.session_state.query_to_send = ""
    if "awaiting_detailed_response" not in st.session_state:
        st.session_state.awaiting_detailed_response = False
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""


def display_chat_history():

    for message in st.session_state.chat_history:
        if message['type'] == 'user':
            st.markdown(
                f'<div class="user-message">👤 {message["content"]}</div>',
                unsafe_allow_html=True
            )
        elif message['type'] == 'bot':
            st.markdown(
                f'<div class="bot-message">🤖 {message["content"]}</div>',
                unsafe_allow_html=True
            )


def handle_query(query, chunks, embeddings, model):
    if not query.strip():
        return

    # Kontrollera om användaren svarar "ja" på frågan om detaljerat svar
    if st.session_state.awaiting_detailed_response and query.lower().strip() in ['ja', 'yes', 'j']:
        st.session_state.chat_history.append({
            'type': 'user',
            'content': query
        })

        with st.spinner("🔍 Skapar detaljerat svar..."):
            detailed_answer = generate_response(
                st.session_state.last_query, chunks, embeddings, model, detailed=True)

        st.session_state.chat_history.append({
            'type': 'bot',
            'content': detailed_answer
        })

        st.session_state.awaiting_detailed_response = False
        st.session_state.last_query = ""

    else:
        st.session_state.chat_history.append({
            'type': 'user',
            'content': query
        })

        with st.spinner("🔍 Söker i Visitas riktlinjer..."):
            answer = generate_response(query, chunks, embeddings, model)

        st.session_state.chat_history.append({
            'type': 'bot',
            'content': answer
        })

        # Sätt flagga för att vänta på detaljerat svar
        if "Vill du ha ett mer detaljerat svar?" in answer:
            st.session_state.awaiting_detailed_response = True
            st.session_state.last_query = query
        else:
            # Återställ flaggan om det inte är en fråga om detaljerat svar
            st.session_state.awaiting_detailed_response = False

    st.session_state.query_to_send = ""


def show_detailed_option():
    """Visa knapp för detaljerat svar"""
    if (st.session_state.chat_history and
        not st.session_state.awaiting_detailed_response and
            len(st.session_state.chat_history) >= 1):

        last_message = st.session_state.chat_history[-1]

        # Visa bara knappen om det senaste meddelandet är ett kort bot-svar
        if (last_message['type'] == 'bot' and
                len(last_message['content']) < 300):

            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("📖 Få detaljerat svar", key=f"detailed_{len(st.session_state.chat_history)}"):
                    # Hämta den ursprungliga frågan
                    if len(st.session_state.chat_history) >= 2:
                        original_query = st.session_state.chat_history[-2]['content']

                        # Lägg till användarens "klick" i chatten
                        st.session_state.chat_history.append({
                            'type': 'user',
                            'content': 'Ja, ge mig ett detaljerat svar'
                        })

                        # Generera detaljerat svar direkt
                        with st.spinner("🔍 Skapar detaljerat svar..."):
                            chunks, embeddings = load_data()
                            api_key = get_api_key()
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel("gemini-1.5-flash")

                            detailed_answer = generate_response(
                                original_query, chunks, embeddings, model, detailed=True)

                        st.session_state.chat_history.append({
                            'type': 'bot',
                            'content': detailed_answer
                        })

                        st.rerun()


def main():
    load_advanced_css()

    api_key = get_api_key()
    if not api_key:
        st.error("❌ **API-nyckel saknas!**")
        st.info("Lägg till API_KEY i .env fil eller Streamlit secrets")
        st.stop()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    chunks, embeddings = load_data()
    initialize_session_state()

    col1, col2, col3 = st.columns([1, 4, 1])

    with col2:
        st.markdown(
            '<h1 class="main-title">🍽️ Livsmedelshygien Expert</h1>',
            unsafe_allow_html=True
        )

        st.markdown(
            '''
            <p class="subtitle">Din AI-guide för säker mat och hygienregler</p>
            <p class="subtitle">Exempel på frågor: Vad är HACCP? Du kan också skriva ett ord, tex fritering</p>
            ''',
            unsafe_allow_html=True
        )

        # Visa chat-historik
        display_chat_history()

        # Visa knapp för detaljerat svar
        show_detailed_option()

        if st.session_state.query_to_send:
            handle_query(st.session_state.query_to_send,
                         chunks, embeddings, model)
            st.rerun()

        with st.form("chat_form", clear_on_submit=True):
            query = st.text_input(
                label="Din fråga:",
                placeholder="T.ex. Vilken temperatur ska kött tillagas vid?",
                label_visibility="collapsed"
            )

            col_send, col_clear = st.columns([3, 1])
            with col_send:
                submitted = st.form_submit_button("🚀 Skicka fråga")
            with col_clear:
                clear_chat = st.form_submit_button("🗑️ Rensa")

        if submitted and query.strip():
            handle_query(query, chunks, embeddings, model)
            st.rerun()

        if clear_chat:
            st.session_state.chat_history = []
            st.rerun()

        with st.expander("ℹ️ Om denna chatbot"):
            st.markdown("""
                    
            Denna AI-chatbot är tränad på Visitas officiella branschriktlinjer för restauranger 
            och hjälper dig med frågor om livsmedelshygien, HACCP, temperaturkontroll och säkra rutiner.
            
            **📚 Datakälla:** Visitas Branschriktlinjer 2021  
            **🤖 AI-teknologi:** Google Gemini + RAG  
            **🎯 Användningsområde:** Utbildning och rådgivning inom livsmedelshygien
            
            **⚠️ Observera:** Detta är ett utbildningsverktyg. För officiell rådgivning, 
            konsultera alltid kvalificerade livsmedelsexperter eller myndigheter.
            """)

        st.markdown("""
        <div style="text-align: center; margin-top: 2rem; color: rgba(255,255,255,0.7); font-size: 0.9rem;">
            Utvecklad med ❤️ för säkrare livsmedelshantering
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
