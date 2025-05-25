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

    /* Svart placeholder-text */
    .stTextInput > div > div > input::placeholder {
        color: black !important;
        opacity: 1 !important;
    }

    /* "Din fråga" label */
    .stTextInput > label {
        font-size: 24px !important;
        font-weight: bold !important;
        color: white !important;
    }

    /* Förstora textfältets textstorlek */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.9);
        border: 2px solid white;
        border-radius: 5px;
        padding: 10px;
        font-size: 1.2rem;
    }

    /* Styling för huvudtitel */
    .main-title {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    /* Styling för svar-sektion */
    .answer-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .answer-container h3 {
        font-size: 1.6rem;
        color: #2596be;
        margin-bottom: 10px;
    }

    .answer-container p {
        font-size: 1.25rem;
        color: #333;
    }

    /* Styling för knappar */
    .stButton > button {
        background-color: white;
        color: rgb(37, 150, 190);
        border: 2px solid white;
        border-radius: 5px;
        font-weight: bold;
        padding: 10px 20px;
        margin: 5px;
        font-size: 1.1rem;
    }

    .stButton > button:hover {
        background-color: rgb(37, 150, 190);
        color: white;
        border: 2px solid white;
    }

    /* Styling för expander */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 5px;
    }

    .stSpinner {
        color: white;
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
            },
            safety_settings={"category": "harassment", "threshold": 3}
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

# === HUVUDAPPLIKATION ===

def main():
    # Ladda CSS
    load_css()
    
    # Konfiguration
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    
    if not API_KEY:
        st.error("API_KEY saknas! Kontrollera din .env-fil.")
        st.stop()
    
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("models/gemini-2.0-flash")

    # Ladda data
    chunks, embeddings = load_data()

    # Titel
    st.markdown('<h1 class="main-title">🍽️ Livsmedelshygien Chatbot</h1>', unsafe_allow_html=True)

    # Initialisera session state
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    if 'last_context' not in st.session_state:
        st.session_state.last_context = ""
    if 'show_detailed_option' not in st.session_state:
        st.session_state.show_detailed_option = False

    # Centrera innehållet och begränsa bredden
    col_left, main_col, col_right = st.columns([1, 3, 1])

    with main_col:
        st.markdown('<p style="font-size: 24px; font-weight: bold; color: white; margin-bottom: 5px;">Din fråga:</p>', unsafe_allow_html=True)
        
        # Skapa input och knapp på samma rad
        input_col, button_col = st.columns([5, 1])
        
        with input_col:
            query = st.text_input("", placeholder="T.ex. Vad är HACCP?", key="query_input", label_visibility="collapsed")
        
        with button_col:
            # Använd CSS för exakt positionering
            st.markdown("""
            <style>
            .send-button {
                margin-top: -8px;
            }
            </style>
            """, unsafe_allow_html=True)
            send_button = st.button("➤", help="Skicka fråga", key="send_btn")

    # Hantera både Enter-tryck och knapp-klick
    if query and (send_button or query != st.session_state.get('previous_query', '')):
        st.session_state.previous_query = query
        
        with st.spinner("🔍 Söker efter relevant information..."):
            try:
                # Sök relevanta chunks
                normalized_query = query.lower().strip()
                relevant_chunks = semantic_search(normalized_query, chunks, embeddings, k=5)

                context = "\n\n".join(relevant_chunks)
                
                # Spara för eventuell detaljerad fråga
                st.session_state.last_query = query
                st.session_state.last_context = context

                # Generera svar
                answer = generate_response(normalized_query, context, model)

                # ✅ Visa svaret i ett korrekt inneslutet block
                st.markdown(f'''
                    <div class="answer-container">
                        <h3>💬 Svar:</h3>
                        <p>{answer}</p>
                    </div>
                ''', unsafe_allow_html=True)

                # Kontrollera om svaret innehåller frågan om detaljerat svar
                if "Vill du ha ett mer detaljerat svar?" in answer:
                    st.session_state.show_detailed_option = True
            
            except Exception as e:
                st.error(f"Ett fel inträffade: {e}")
                logging.error(f"Fel i huvudloop: {e}")


    # Visa detaljerad svar-knapp om det behövs
    if st.session_state.show_detailed_option:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📋 Ja, ge mer detaljerat svar", key="detailed_button"):
                with st.spinner("📝 Skapar detaljerat svar..."):
                    try:
                        detailed_answer = generate_response(
                            st.session_state.last_query.lower().strip(),

                            st.session_state.last_context, 
                            model, 
                            detailed=True
                        )

                        if not detailed_answer:
                            detailed_answer = "⚠️ Inget svar kunde genereras."

                        # Visa svaret i ett vitt block
                        st.markdown(f"""
                            <div class="answer-container">
                                <h3>📋 Detaljerat svar:</h3>
                                <p>{detailed_answer}</p>
                            </div>
                        """, unsafe_allow_html=True)

                        st.session_state.show_detailed_option = False

                    except Exception as e:
                        st.error(f"Ett fel inträffade: {e}")
                        logging.error(f"Fel vid generering av detaljerat svar: {e}")


    # Info om systemet
    with st.expander("ℹ️ Om denna chatbot"):
        st.markdown("""
        **Livsmedelshygien Chatbot**
        
        - Denna chatbot är byggd för att svara på frågor om livsmedelshygien
        - Svaren baseras på Visitas dokumentation om branschriktlinjer
        - Källa: [Visitas Branschriktlinjer](https://visita.se/app/uploads/2021/06/Visita_Branschriktlinjer-print_2021.pdf)
        
        **Så här använder du chatten:**
        1. Skriv din fråga i textfältet
        2. Tryck Enter eller klicka på pil-knappen (➤)
        3. Om du vill ha mer detaljerad information, klicka på "Ja, ge mer detaljerat svar"
        """)

if __name__ == "__main__":
    main()