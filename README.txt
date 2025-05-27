📄 README: Så här kör du din kontextstyrda chatbot i Streamlit

1. Se till att dessa filer finns i samma mapp:
   - streamlit_app.py
   - chunks.pkl
   - embeddings_chunks.pkl
   - .env (din API-nyckel)

2. Skapa .env-filen om den inte finns:
   API_KEY=ditt_api_key_här

3. Installera nödvändiga paket:
   pip install streamlit google-generativeai python-dotenv

4. Starta chatbotten:
   streamlit run streamlit_app.py
   eller https://chatbotlivsmedelshygien.streamlit.app/

✅ Appen laddar in dina chunks och embeddings, söker efter relevant innehåll, skickar det till modellen och visar svaret i ett webbaserat gränssnitt.
