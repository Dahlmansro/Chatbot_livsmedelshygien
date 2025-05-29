
# 🍽️ Livsmedelshygien Chatbot

Ett AI-stöd för personer som arbetar med livsmedelshygien i restaurangkök. Genom att använda en chattbot som bygger på generativ AI ges användare möjlighet att snabbt och enkelt få svar på vanliga frågor – baserat på officiella branschriktlinjer.

## Syfte

Detta projekt syftar till att underlätta förståelsen av livsmedelshygien för restaurangpersonal. Projektet bygger på branschorganisationens publika dokumentation och innebär därför inga affärsmässiga eller etiska risker.

📄 Dokumentationen omfattar cirka 60 sidor och kan vara svår att ta till sig för personer i köksmiljö. 
https://visita.se/app/uploads/2021/05/Visita_Branschriktlinjer2021_uppslag-webb.pdf 
💬 Med chattboten kan användare ställa frågor och få enkla, snabba svar – samt klicka fram mer detaljerad information vid behov.

Du kan testa chatboten via följande länk:
https://chatbotlivsmedelshygien.streamlit.app/

## Funktioner

- Textbaserad fråga-svar-funktion via chatbot
- Snabbsvar + möjlighet till mer detaljerad information
- Dokumentindata (branschriktlinjer) hanteras med semantisk chunking
- Embedding + likhetssökning för att hitta relevanta avsnitt i texten
- Gemini API (Google Generative AI) används för generering av svar

## Använda paket

### I app (`streamlit_app.py`)
```python
import streamlit as st
import pickle
import os
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
```

### I notebook (`Semantic_chunking_1.ipynb`)
```python
import os, time, pickle, json, re, logging
from typing import List, Any

import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
nltk.download('punkt')
nltk.download('punkt_tab')

from sklearn.metrics.pairwise import cosine_similarity
import fitz  # pymupdf

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
```

## Utvärdering

Modellen har testats mot riktlinjedokumentet från Visita och fungerar tillfredsställande som stöd i vardagliga frågor. Exempel på frågor som modellen svarar bra på:

- "Vad gäller för kylförvaring?"
- "Måste jag ha dokumenterade rutiner för rengöring?"

## Potentiell vidareutveckling

- ➕ Addera fler källor, t.ex. dokumentation från Livsmedelsverket (kräver vektordatabas)
- 🌍 Stöd för flera språk eller skråkstöd för personer med utländsk bakgrund
- 🎤 Stöd för röstinput
- 🧩 Bättre moduluppdelning av kod – dela upp i fler filer
- 📦 Paketera projektet som en installerbar applikation

## 📁 Struktur

```plaintext
.
├── README.md                      # Du läser den nu
├── requirements.txt               # Alla beroenden
├── Semantic_chunking_1.ipynb      # Notebook för chunkning & embeddings
├── streamlit_app.py               # Själva applikationen
├── semantic_chunks.pkl            # Förberedda chunks
├── komponent_utvardering.*        # Utvärderingsdata
├── Utvärdering_modell.txt         # Resultat från modellutvärdering
└── Visita_Branschriktlinjer*.pdf  # Källdokument
```

## Kom igång

1. Klona repot
2. Lägg in API-nyckeln i `.env`:
   ```
   API_KEY=din_google_api_nyckel
   ```
3. Installera beroenden:
   ```
   pip install -r requirements.txt
   ```
4. Kör appen:
   ```
   streamlit run streamlit_app.py
   ```
