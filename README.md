# AIGmailChatbot 

An intelligent Gmail assistant powered by **Google Gemini**, **RAG (Retrieval-Augmented Generation)**, and **Streamlit**.  
Ask natural-language questions about your inbox and get smart, context-aware answers instantly. The app supports chat-based RAG Q&A, email summarization, email categorization (work / personal / finance / spam), and a combined summary+category analysis mode.

---

## Features
- **Semantic email search** using Sentence-Transformer embeddings + ChromaDB  
- **AI-powered answers & summarization** using Google Gemini 2.5 Flash via a RAG pipeline  
- **Chat-like Streamlit UI** with modes: Chat (RAG), Summarize, Categorize, Both (Summary + Category)  
- **Summarize emails** (2–4 sentence abstractive summaries)  
- **Categorize emails** into: `work`, `personal`, `finance`, `spam` (strong rules + Gemini fallback)  
- **Caching** for summaries & categories to reduce API calls and preserve quota  
- Persistent vector memory using Chroma (email embeddings)  
- CLI chatbot (`rag_email_agent.py`) included for terminal use

---

## Prerequisites
- Python **3.9+**  
- Gmail API OAuth credentials (`client_secret.json`) with Gmail API enabled  
- Gemini API key (`GEMINI_API_KEY`) in a `.env` file  
- (Optional) GPU for faster embedding—otherwise CPU is used

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/aleenavarghese29/AIGmailChatbot.git
cd AIGmailChatbot
````

### 2. Create & activate virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Credentials

* Place `client_secret.json` in the project root.
* Create a `.env` file:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Usage

### Run Streamlit Web App (recommended)

```bash
streamlit run chatbot.py
```


### Run CLI Chatbot

```bash
python rag_email_agent.py
```

---

## Architecture (How it works)

```
Gmail API -> gmail_api.py -> Clean Email Text
             ↓
      HuggingFace Embeddings (Sentence Transformers)
             ↓
         Chroma Vector DB (persisted embeddings)
             ↓
      Retrieval (semantic similarity search)
             ↓
      Gemini (generation & summarization)  <-- Cached results used when possible
             ↓
       Streamlit UI / CLI outputs
```

---

## Project Structure

```
AIGmailChatbot/
├── chatbot.py                # Streamlit web app (main UI)
├── rag_email_agent.py        # Terminal-based chatbot
├── gmail_api.py              # Gmail API logic & email extraction
├── google_apis.py            # OAuth helper (tokens)
├── requirements.txt
├── email_memory/             # Chroma vector DB (persist directory)
├── token_files/              # Gmail OAuth tokens (do not commit)
├── .env                      # Environment variables (GEMINI_API_KEY)
├── client_secret.json        # Gmail OAuth credentials (do not commit)
└── README.md                 # This file
```

---

## Gitignore

**Never commit** sensitive files:

```
.env
client_secret.json
token_files/
email_memory/
venv/
__pycache__/
```

---

## 🧾 Author

**Author:** Aleena Varghese
GitHub: [https://github.com/aleenavarghese29/AIGmailChatbot](https://github.com/aleenavarghese29/AIGmailChatbot)

