import streamlit as st
import sys, os
sys.path.append(r"G:/ai_email_project")  # Make sure your gmail_api.py is in this folder

from gmail_api import init_gmail_service, get_email_messages, get_email_message_details
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from dotenv import load_dotenv
load_dotenv()  
# ----------------- Setup -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.title("Generative AI Email Assistant")
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

def classify_email(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=384, padding=True, truncation=True)
    inputs = {k: v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = predictions.argmax().item()
        confidence = predictions[0][predicted_class_id].item()
    return LABELS[predicted_class_id], confidence

URGENT_KEYWORDS = ["urgent", "asap", "important", "immediate", "action required"]

def detect_urgency(text):
    text_lower = text.lower()
    for keyword in URGENT_KEYWORDS:
        if keyword in text_lower:
            return True
    return False

# ----------------- Chatbot Input -----------------
query = st.text_input("Type your query here (e.g., Show urgent emails, Summarize finance emails)")

if st.button("Submit") and query:
    st.write(f"**You asked:** {query}")
    query_lower = query.lower()  
    
    # Fetch latest emails (max 20 for speed)
    messages = get_email_messages(service, max_results=20)
    
    # Store filtered results
    filtered_emails = []

    # Handle urgent emails
    if "urgent" in query.lower():
        for msg in messages:
            details = get_email_message_details(service, msg['id'])
            if detect_urgency(details['body']):
                filtered_emails.append(details)
        if not filtered_emails:
            st.write("No urgent emails found.")
        else:
            st.write(f"Found {len(filtered_emails)} urgent emails:")
            for e in filtered_emails:
                st.write(f"- **Subject:** {e['subject']} | **From:** {e['sender']}")
    
    # Handle finance email summaries
    elif "finance" in query.lower() and "summarize" in query.lower():
        count = 0
        for msg in messages:
            details = get_email_message_details(service, msg['id'])
            category, _ = classify_email(details['body'])
            if category == "updates" or "finance" in details['subject'].lower():
                filtered_emails.append(details)
        if not filtered_emails:
            st.write("No finance emails found.")
        else:
            st.write(f"Summaries of last {min(5,len(filtered_emails))} finance emails:")
            for e in filtered_emails[:5]:
                try:
                    summary = summarizer(
                        e['body'], max_length=120, min_length=30, do_sample=False
                    )[0]['summary_text']
                except:
                    summary = e['body'][:200]
                st.write(f"- **Subject:** {e['subject']} | **Summary:** {summary}")

    # Handle marking as spam
    
    elif "spam" in query_lower:
        results = []  # <-- define the list first
        for msg in messages:
            details = get_email_message_details(service, msg['id'])
            if "spam" in details['subject'].lower() or "spam" in details['body'].lower():
                results.append(details)
        if results:
            st.write("Found the following potential spam emails:")
            for e in results:
                st.write(f"- **Subject:** {e['subject']} | **From:** {e['sender']}")
        else:
            st.write("No spam emails found.")
