import streamlit as st
import sys, os
sys.path.append(r"G:/ai_email_project")  # Path to your gmail_api.py

from gmail_api import init_gmail_service, get_email_messages, get_email_message_details
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from dotenv import load_dotenv
load_dotenv() 
# ----------------- Setup -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.title("AI Gmail Chatbot")
st.write("Ask me about your emails!")

# --- Step 1: Initialize Gmail API ---
client_file = "client_secret.json"
service = init_gmail_service(client_file)
st.success("Gmail API initialized successfully.")

# --- Step 2: Load Models ---
st.info("Loading models, please wait...")
CLASSIFIER_MODEL_ID = "jason23322/email-classifier-distilbert"
hf_token = os.getenv("HF_TOKEN")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    CLASSIFIER_MODEL_ID, token=hf_token
).to(device)

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0 if device.type=='cuda' else -1
)
st.success("Models loaded successfully!")

# --- Classification & Urgency Functions ---
LABELS = ["forum", "promotions", "social_media", "spam", "updates", "verify_code"]
URGENT_KEYWORDS = ["urgent", "asap", "important", "immediate", "action required"]

def classify_email(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=384, padding=True, truncation=True)
    inputs = {k: v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = predictions.argmax().item()
        confidence = predictions[0][predicted_class_id].item()
    return LABELS[predicted_class_id], confidence

def detect_urgency(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in URGENT_KEYWORDS)

# ----------------- Chatbot Input -----------------
query = st.text_input("Type your query (e.g., Show urgent emails, Summarize emails, Search by sender, Keyword search)")

if st.button("Submit") and query:
    st.write(f"**You asked:** {query}")
    query_lower = query.lower()
    
    # Fetch latest emails (max 50 for better results)
    messages = get_email_messages(service, max_results=50)

    # ---------------- Handle Urgent Emails ----------------
    if "urgent" in query_lower:
        urgent_emails = []
        for msg in messages:
            details = get_email_message_details(service, msg['id'])
            if detect_urgency(details['body']):
                urgent_emails.append(details)
        if urgent_emails:
            st.write(f"Found {len(urgent_emails)} urgent emails:")
            for e in urgent_emails:
                st.write(f"- **Subject:** {e['subject']} | **From:** {e['sender']}")
        else:
            st.write("No urgent emails found.")

   # ---------------- Handle Email Summaries ----------------
    elif "summarize" in query_lower:
        emails_to_summarize = []
        for msg in messages:
            details = get_email_message_details(service, msg['id'])
            emails_to_summarize.append(details)
        if emails_to_summarize:
            st.write(f"Summaries of last {min(5, len(emails_to_summarize))} emails:")
            for e in emails_to_summarize[:5]:
                try:
                    summary = summarizer(
                        e['body'], max_length=120, min_length=30, do_sample=False
                    )[0]['summary_text']
                except:
                    summary = e['body'][:200] + "..."  # fallback
                st.write(f"- **Subject:** {e['subject']} | **Summary:** {summary}")
        else:
            st.write("No emails found to summarize.")


    # ---------------- Handle Spam Detection ----------------
    elif "spam" in query_lower:
        spam_emails = []
        for msg in messages:
            details = get_email_message_details(service, msg['id'])
            if "spam" in details['subject'].lower() or "spam" in details['body'].lower():
                spam_emails.append(details)
        if spam_emails:
            st.write("Found the following potential spam emails:")
            for e in spam_emails:
                st.write(f"- **Subject:** {e['subject']} | **From:** {e['sender']}")
        else:
            st.write("No spam emails found.")
    # ---------------- Search by Sender ----------------
    elif "from:" in query_lower:
        parts = query_lower.split("from:")
        if len(parts) < 2 or not parts[1].strip():
            st.write("Please provide the sender email after 'from:'. Example: from:example@gmail.com")
        else:
            sender_query = parts[1].strip()
            matched_emails = []
            for msg in messages:
                details = get_email_message_details(service, msg['id'])
                if sender_query in details['sender'].lower():
                    matched_emails.append(details)
            if matched_emails:
                st.write(f"Emails from {sender_query}:")
                for e in matched_emails:
                    st.write(f"- **Subject:** {e['subject']} | **From:** {e['sender']}")
            else:
                st.write(f"No emails found from {sender_query}.")
    # ---------------- Keyword Search ----------------
    elif "search:" in query_lower:
        keyword = query_lower.split("search:")[1].strip()
        keyword_emails = []
        for msg in messages:
            details = get_email_message_details(service, msg['id'])
            if keyword in details['subject'].lower() or keyword in details['body'].lower():
                keyword_emails.append(details)
        if keyword_emails:
            st.write(f"Emails containing '{keyword}':")
            for e in keyword_emails:
                st.write(f"- **Subject:** {e['subject']} | **From:** {e['sender']}")
        else:
            st.write(f"No emails found containing '{keyword}'.")

    else:
        st.write("Sorry, I didn't understand your query. You can ask about urgent emails, finance summaries, spam detection, search by sender (from:email), or keyword search (search:keyword).")
