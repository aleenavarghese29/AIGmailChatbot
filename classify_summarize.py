from gmail_api import init_gmail_service, get_email_messages, get_email_message_details
from transformers import pipeline,DistilBertTokenizer, DistilBertForSequenceClassification
import torch, os
from dotenv import load_dotenv
load_dotenv()  
# --- Step 0: Check for CUDA (GPU) availability ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Step 1: Initialize Gmail API ---
client_file = "client_secret.json"
service = init_gmail_service(client_file)
print("Gmail API initialized successfully.")

# --- Step 2: Load Hugging Face Models ---

print("Loading Hugging Face models...")
CLASSIFIER_MODEL_ID = "jason23322/email-classifier-distilbert"
hf_token = os.getenv("HF_TOKEN")
# Use the tokenizer from 'distilbert-base-uncased' instead of the gated repo
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(CLASSIFIER_MODEL_ID, token=hf_token)

# Summarizer (pipeline handles device automatically)
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0 if device.type == 'cuda' else -1
)
print("Models loaded successfully.")

# --- Step 2b: Define Classification Function ---
LABELS = ["forum", "promotions", "social_media", "spam", "updates", "verify_code"]

def classify_email(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=384, padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = predictions.argmax().item()
        confidence = predictions[0][predicted_class_id].item()
    
    return LABELS[predicted_class_id], confidence

# --- Step 3: Fetch latest emails ---
messages = get_email_messages(service, max_results=5)

# --- Step 4: Process emails ---
for msg in messages:
    details = get_email_message_details(service, msg['id'])
    
    full_email_text = f"SUBJECT: {details['subject']}. BODY: {details['body']}"

    # Summarize email
    try:
        summary_output = summarizer(
            full_email_text, max_length=120, min_length=30, do_sample=False, truncation=True
        )
        summary = summary_output[0]['summary_text']
    except Exception as e:
        summary = full_email_text[:200].replace('\n', ' ')  # fallback
        print(f"Warning: Summarization failed for email {details['subject']}. Error: {e}")

    # Classify email
    try:
        category, confidence = classify_email(summary)
    except Exception as e:
        category, confidence = "Unknown", 0.0
        print(f"Warning: Classification failed for email {details['subject']}. Error: {e}")

    # Print results
    print("--------------------------------------------------")
    print(f"Subject : {details['subject']}")
    print(f"From    : {details['sender']}")
    print(f"Category: {category} (Confidence: {confidence:.3f})")
    print(f"Summary : {summary}")
    print("--------------------------------------------------\n")
