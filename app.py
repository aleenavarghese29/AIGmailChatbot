import streamlit as st
import sys, os, json
import torch
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification

# Add local path if needed
sys.path.append("./")

# Streamlit-specific Gmail init
from gmail_api_streamlit import init_gmail_service_streamlit
from gmail_api import get_email_messages, get_email_message_details

# ----------------- Setup -----------------
st.title("AI Gmail Chatbot")
st.write("Ask me about your emails!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Gmail API Init -----------------
if "gcp_oauth" not in st.secrets:
    st.error("❌ Missing Gmail API credentials in Streamlit secrets!")
    st.stop()

client_config = st.secrets["gcp_oauth"]  

# Initialize Gmail service for Streamlit
service = init_gmail_service_streamlit(client_config)
st.success("✅ Gmail API initialized successfully.")




# ----------------- Hugging Face Models -----------------
st.info("Loading models, please wait...")

CLASSIFIER_MODEL_ID = "jason23322/email-classifier-distilbert"

# Hugging Face token from Streamlit secrets
if "huggingface" not in st.secrets or "token" not in st.secrets["huggingface"]:
    st.error("❌ Missing Hugging Face token in Streamlit secrets!")
    st.stop()

hf_token = st.secrets["huggingface"]["token"]

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    CLASSIFIER_MODEL_ID, use_auth_token=hf_token
).to(device)

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    use_auth_token=hf_token,
    device=0 if device.type == "cuda" else -1
)

st.success("✅ Models loaded successfully!")

# ----------------- Classification & Urgency -----------------
LABELS = ["forum", "promotions", "social_media", "spam", "updates", "verify_code"]
URGENT_KEYWORDS = ["urgent", "asap", "important", "immediate", "action required"]

def classify_email(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=384, padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = predictions.argmax().item()
        confidence = predictions[0][predicted_class_id].item()
    return LABELS[predicted_class_id], confidence

def detect_urgency(text):
    return any(keyword in text.lower() for keyword in URGENT_KEYWORDS)

# ----------------- Chatbot Input -----------------
query = st.text_input("Type your query (e.g., Show urgent emails, Summarize emails, Search by sender, Keyword search)")

if st.button("Submit") and query:
    st.write(f"**You asked:** {query}")
    query_lower = query.lower()

    messages = get_email_messages(service, max_results=50)

    # Urgent emails
    if "urgent" in query_lower:
        urgent_emails = []
        for msg in messages:
            details = get_email_message_details(service, msg["id"])
            if detect_urgency(details["body"]):
                urgent_emails.append(details)
        if urgent_emails:
            st.write(f"Found {len(urgent_emails)} urgent emails:")
            for e in urgent_emails:
                st.write(f"- **Subject:** {e['subject']} | **From:** {e['sender']}")
        else:
            st.write("No urgent emails found.")

    # Summarization
    elif "summarize" in query_lower:
        emails_to_summarize = [get_email_message_details(service, msg["id"]) for msg in messages]
        if emails_to_summarize:
            st.write(f"Summaries of last {min(5, len(emails_to_summarize))} emails:")
            for e in emails_to_summarize[:5]:
                try:
                    summary = summarizer(e["body"], max_length=120, min_length=30, do_sample=False)[0]["summary_text"]
                except:
                    summary = e["body"][:200] + "..."
                st.write(f"- **Subject:** {e['subject']} | **Summary:** {summary}")
        else:
            st.write("No emails found to summarize.")

    # Spam detection
    elif "spam" in query_lower:
        spam_emails = []
        for msg in messages:
            details = get_email_message_details(service, msg["id"])
            if "spam" in details["subject"].lower() or "spam" in details["body"].lower():
                spam_emails.append(details)
        if spam_emails:
            st.write("Found potential spam emails:")
            for e in spam_emails:
                st.write(f"- **Subject:** {e['subject']} | **From:** {e['sender']}")
        else:
            st.write("No spam emails found.")

    # Search by sender
    elif "from:" in query_lower:
        sender_query = query_lower.split("from:")[1].strip()
        matched_emails = [
            get_email_message_details(service, msg["id"])
            for msg in messages
            if sender_query in get_email_message_details(service, msg["id"])["sender"].lower()
        ]
        if matched_emails:
            st.write(f"Emails from {sender_query}:")
            for e in matched_emails:
                st.write(f"- **Subject:** {e['subject']} | **From:** {e['sender']}")
        else:
            st.write(f"No emails found from {sender_query}.")

    # Keyword search
    elif "search:" in query_lower:
        keyword = query_lower.split("search:")[1].strip()
        keyword_emails = [
            get_email_message_details(service, msg["id"])
            for msg in messages
            if keyword in get_email_message_details(service, msg["id"])["subject"].lower()
            or keyword in get_email_message_details(service, msg["id"])["body"].lower()
        ]
        if keyword_emails:
            st.write(f"Emails containing '{keyword}':")
            for e in keyword_emails:
                st.write(f"- **Subject:** {e['subject']} | **From:** {e['sender']}")
        else:
            st.write(f"No emails found containing '{keyword}'.")

    else:
        st.write("❓ I didn't understand your query. Try: urgent emails, summarize, spam, from:email, or search:keyword.")
