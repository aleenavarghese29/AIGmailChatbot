import streamlit as st
from dotenv import load_dotenv
import os

# --- Gmail Utilities ---
from gmail_api import init_gmail_service, get_email_messages, get_email_message_details

# --- LangChain Imports ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai

# Load environment variables
load_dotenv()

# --- Cache the embeddings and vector DB to avoid reloading on every rerun ---
@st.cache_resource
def setup_embeddings_and_db():
    """Initialize embeddings and vector database once, cached for the session."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    
    vector_db = Chroma(
        collection_name="emails",
        embedding_function=embeddings,
        persist_directory="./email_memory"
    )
    return embeddings, vector_db

# --- Gmail API Setup ---
client_file = "client_secret.json"
service = init_gmail_service(client_file)
print("✅ Gmail API initialized successfully.")

# --- Vector DB Setup (cached) ---
embeddings, vector_db = setup_embeddings_and_db()
print("✅ Vector DB initialized (Chroma).")

# --- Email Loading Function ---
def load_emails_to_memory(service, max_results=15):
    print(f"📩 Fetching last {max_results} emails...")
    messages = get_email_messages(service, max_results=max_results)

    for msg in messages:
        details = get_email_message_details(service, msg["id"])
        subject, body, sender = details["subject"], details["body"], details["sender"]

        content = f"Subject: {subject}\nFrom: {sender}\nBody: {body}"
        vector_db.add_texts([content], metadatas=[{"subject": subject, "from": sender}])

    vector_db.persist()
    print(f"✅ Stored {len(messages)} emails in Chroma memory.")


load_emails_to_memory(service, max_results=15)

# --- Cache Gemini model setup ---
@st.cache_resource
def setup_gemini():
    """Initialize Gemini model once, cached for the session."""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("❌ GEMINI_API_KEY not found in environment. Please add it to your .env file.")
    
    genai.configure(api_key=gemini_api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

gemini_model = setup_gemini()

SYSTEM_INSTRUCTIONS = (
    "You are an AI assistant that answers questions based only on the provided email context. "
    "If you cannot find the answer in the context, politely state that the information "
    "is not available in the emails you have access to. Answer concisely."
)


# --- RAG Function ---
def answer_query_via_gemini(query, k=10, temperature=0.3, max_tokens=512):
    """
    Retrieve relevant emails from Chroma and generate an answer using Google Gemini.
    
    Args:
        query: User question
        k: Number of top results to retrieve (default 10 for better context)
        temperature: Model temperature (0.0-2.0 for Gemini)
        max_tokens: Maximum response length (default 512 for concise answers)
    
    Returns:
        Answer string from Gemini or error message
    """
    # Retrieve top-k relevant email documents from Chroma
    try:
        docs = vector_db.similarity_search(query, k=k)
    except Exception:
        # Fallback to retriever if similarity_search isn't available
        retriever = vector_db.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)

    if not docs:
        return "No relevant emails found in memory."

    # Build context string from retrieved docs
    context = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        context.append(
            f"Subject: {meta.get('subject', 'N/A')}\n"
            f"From: {meta.get('from', 'N/A')}\n\n"
            f"{d.page_content}"
        )
    context_str = "\n\n---\n\n".join(context)

    user_message = (
        f"You are a helpful email assistant. Answer the user's question based on the provided email context. "
        f"Be concise and direct. Include specific details (sender, subject, key content) only when relevant. "
        f"Use simple, natural language—avoid lengthy explanations.\n\n"
        f"Email Context:\n{context_str}\n\n"
        f"User question: {query}\n"
        f"Answer concisely. If the information is not in the emails, simply say you don't know."
    )

    # Call Gemini API
    try:
        response = gemini_model.generate_content(
            user_message,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"Gemini API request failed: {e}"


# --- Streamlit UI ---
st.set_page_config(page_title="Email Assistant Chatbot", page_icon="📧", layout="centered")

st.markdown(
    """
    <h2 style='text-align:center;'>📧 Email Assistant Chatbot</h2>
    <p style='text-align:center; font-size:18px;'>
    Hey there! 😊 Got a minute? Do you need to know something from your emails?<br>
    I'm here to help you anytime!
    </p>
    """,
    unsafe_allow_html=True
)

# Session messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Show chat history
for role, text in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"🧑 **You:** {text}")
    else:
        st.markdown(f"🤖 **Agent:** {text}")


# User Input
query = st.text_input("Ask me anything about your emails:")

if st.button("Send"):
    if query.strip():
        st.session_state.chat_history.append(("user", query))

        # Get answer from your existing function
        answer = answer_query_via_gemini(query)

        st.session_state.chat_history.append(("assistant", answer))
        st.rerun()

# Clear chat
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

st.write("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ using Gemini + RAG + Streamlit</p>", unsafe_allow_html=True)
