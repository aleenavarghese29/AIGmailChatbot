# AIGmailChatbot 📧  
An intelligent Gmail assistant powered by **Google Gemini**, **RAG (Retrieval-Augmented Generation)**, and **Streamlit**.  
Ask natural-language questions about your inbox and get smart, context-aware answers instantly.

---

## 🚀 Features
- 🔍 **Semantic email search** using embeddings + ChromaDB  
- 🤖 **AI-powered answers** using Google Gemini 2.5 Flash  
- 💬 **Chat-like Streamlit UI**  
- 📩 Retrieve and summarize emails by sender, keyword, or topic  
- 📊 Extract useful information (senders, dates, content)  
- 💾 Persistent vector memory using Chroma  
- 🖥️ CLI chatbot (`rag_email_agent.py`) included  

### Example Queries
```

Any urgent emails?
What did Aleena say?
Summarize my last 5 emails.
Any messages from John?

````

---

## 📋 Prerequisites
- Python **3.9+**
- Gmail API OAuth credentials (`client_secret.json`)
- Gemini API key (`GEMINI_API_KEY`)
- Gmail account with API enabled

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/aleenavarghese29/AIGmailChatbot.git
cd AIGmailChatbot
````

### 2️⃣ Create Virtual Environment

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

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Credentials

Place your Gmail API credentials here:

```
client_secret.json
```

Create a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

---

## 💻 Usage

### 🌐 Option 1 — Streamlit Web App (Recommended)

```bash
streamlit run chatbot.py
```

Then open: **[http://localhost:8501](http://localhost:8501)**

### 🖥️ Option 2 — Terminal Chatbot

```bash
python rag_email_agent.py
```

---

## 🧠 Architecture (How It Works)

```
Gmail API → gmail_api.py → Clean Email Text
           ↓
Chroma Vector DB (stores and retrieves emails)
           ↓
HuggingFace Embeddings (semantic similarity search)
           ↓
Google Gemini AI (generates answers based on context)
           ↓
Streamlit UI / CLI Chatbot
```

---

## 📁 Project Structure

```
AIGmailChatbot/
├── chatbot.py              # Streamlit web app
├── rag_email_agent.py      # Terminal chatbot
├── gmail_api.py            # Gmail API + email body extraction
├── requirements.txt        # Dependencies
├── email_memory/           # Chroma database
├── token_files/            # Gmail OAuth tokens
├── .env                    # API keys (ignored in git)
├── client_secret.json      # Gmail OAuth credentials
└── README.md               # Documentation
```

---

## 🔒 Security Notice

⚠️ **Never upload these files to GitHub:**

```
.env
client_secret.json
token_files/
email_memory/
venv/
__pycache__/
```

---

## 📦 Key Dependencies

* streamlit
* google-generativeai
* google-api-python-client
* google-auth-oauthlib
* sentence-transformers
* chromadb
* langchain / langchain-community
* torch
* python-dotenv

---

## 🧩 Future Enhancements

* [ ] Email classification (urgent, spam, promotions)
* [ ] Sentiment analysis
* [ ] Attachment preview
* [ ] Multi-language support
* [ ] Auto-reply suggestions
* [ ] Save chat history

---

## 👤 Author

**Aleena Varghese**
GitHub: [https://github.com/aleenavarghese29/AIGmailChatbot](https://github.com/aleenavarghese29/AIGmailChatbot)

---


