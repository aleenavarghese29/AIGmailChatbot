
from dotenv import load_dotenv
import os
load_dotenv()

# --- Gmail Utilities ---
from gmail_api import init_gmail_service, get_email_messages, get_email_message_details

# --- LangChain Imports ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai

# --- Step 1: Initialize Gmail API ---
client_file = "client_secret.json"
service = init_gmail_service(client_file)
print("✅ Gmail API initialized successfully.")

# --- Step 2: Initialize Chroma Vector Store ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
vector_db = Chroma(
    collection_name="emails",
    embedding_function=embeddings,
    persist_directory="./email_memory"
)
print("✅ Vector DB initialized (Chroma).")

# --- Step 3: Fetch and Store Emails ---
def load_emails_to_memory(service, max_results=15):
    print(f"📩 Fetching last {max_results} emails...")
    messages = get_email_messages(service, max_results=max_results)
    for msg in messages:
        details = get_email_message_details(service, msg["id"])
        subject, body, sender = details["subject"], details["body"], details["sender"]

        # Combine subject + body as searchable text
        content = f"Subject: {subject}\nFrom: {sender}\nBody: {body}"
        vector_db.add_texts([content], metadatas=[{"subject": subject, "from": sender}])

    vector_db.persist()
    print(f"✅ Stored {len(messages)} emails in Chroma memory.")

load_emails_to_memory(service, max_results=15)

# --- Step 4: Setup Gemini Client and RAG Function ---

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in environment. Please add it to your .env file.")

genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

SYSTEM_INSTRUCTIONS = (
    "You are an AI assistant that answers questions based only on the provided email context. "
    "If you cannot find the answer in the context, politely state that the information "
    "is not available in the emails you have access to. Answer concisely."
)

def answer_query_via_gemini(query, k=4, temperature=0.3, max_tokens=512):
    """
    Retrieve relevant emails from Chroma and generate an answer using Google Gemini.
    
    Args:
        query: User question
        k: Number of top results to retrieve
        temperature: Model temperature (0.0-2.0 for Gemini)
        max_tokens: Maximum response length
    
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
        f"You are an AI assistant that answers questions based only on the provided email context. "
        f"If you cannot find the answer in the context, politely state that the information "
        f"is not available in the emails.\n\n"
        f"Email Context:\n{context_str}\n\n"
        f"User question: {query}\n"
        f"Provide a concise answer based only on the context. If not present, say you don't know."
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


# --- Step 5: Conversational Query Loop ---
print("\n💬 Hey there! 😊 Are you busy? Do you need to know about your emails?")
print("Check in — I'm here to help! (type 'exit' or 'quit' to stop)\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Have a nice day! Exiting AI Email Agent.")
        break
    answer = answer_query_via_gemini(query)
    print(f"🤖 Agent: {answer}\n")  

