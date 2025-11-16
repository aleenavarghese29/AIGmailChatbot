import streamlit as st
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import re
import hashlib

# --- Gmail Utilities ---
from gmail_api import init_gmail_service, get_email_messages, get_email_message_details

# --- LangChain Imports ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai

# Load environment variables
load_dotenv()

# CACHING LAYER - Prevents redundant API calls


SUMMARY_CACHE = {}
CATEGORY_CACHE = {}

def get_content_hash(content):
    """Generate a hash for email content to use as cache key."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()
def get_content_hash(content):
    """Generate a hash for email content to use as cache key."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

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
print("Gmail API initialized successfully.")

# --- Vector DB Setup (cached) ---
embeddings, vector_db = setup_embeddings_and_db()
print("Vector DB initialized (Chroma).")

# --- Email Loading Function ---
def load_emails_to_memory(service, max_results=15):
    print(f"Fetching last {max_results} emails...")
    messages = get_email_messages(service, max_results=max_results)

    for msg in messages:
        details = get_email_message_details(service, msg["id"])
        subject, body, sender = details["subject"], details["body"], details["sender"]

        content = f"Subject: {subject}\nFrom: {sender}\nBody: {body}"
        vector_db.add_texts([content], metadatas=[{"subject": subject, "from": sender}])

    vector_db.persist()
    print(f"Stored {len(messages)} emails in Chroma memory.")


load_emails_to_memory(service, max_results=15)

# --- Cache Gemini model setup ---
@st.cache_resource
def setup_gemini():
    """Initialize Gemini model once, cached for the session."""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment. Please add it to your .env file.")
    
    genai.configure(api_key=gemini_api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

gemini_model = setup_gemini()

#SUMMARIZATION


def truncate_email_content(content, max_chars=2000):
    """
    Truncate email content to prevent token limit issues.
    Keeps subject and beginning of body.
    """
    lines = content.split('\n')
    subject_line = lines[0] if lines else ""
    
    if len(content) <= max_chars:
        return content
    
    # Keep subject + truncated body
    body_start = content.find('\n', content.find('Body:'))
    if body_start == -1:
        return content[:max_chars]
    
    body = content[body_start:body_start + max_chars - len(subject_line)]
    return subject_line + body + "\n[... content truncated ...]"


def extract_key_sentences(text, num_sentences=5):
    """
    Fallback summarizer: Extract first few sentences from email body.
    """
    # Find body section
    body_start = text.find('Body:')
    if body_start == -1:
        body_text = text
    else:
        body_text = text[body_start + 5:].strip()
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', body_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return "No summary available."
    
    # Return first N sentences
    summary = '. '.join(sentences[:num_sentences])
    if summary and not summary.endswith('.'):
        summary += '.'
    
    return summary[:300]  # Max 300 chars


def safe_extract_gemini_text(response):
    """
    Safely extract text from Gemini response with multiple fallback methods.
    """
    if not response:
        return None
    
    # Method 1: Direct text attribute
    try:
        if hasattr(response, 'text') and response.text:
            return response.text.strip()
    except:
        pass
    
    # Method 2: Candidates and parts
    try:
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content'):
                content = candidate.content
                if hasattr(content, 'parts') and content.parts:
                    text = ''.join(part.text for part in content.parts if hasattr(part, 'text'))
                    if text:
                        return text.strip()
    except:
        pass
    
    # Method 3: Check finish_reason
    try:
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
                # finish_reason: 1=STOP (normal), 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION
                if finish_reason in [2, 3, 4]:
                    return None  # Cannot extract, need fallback
    except:
        pass
    
    return None


def summarize_email(content, max_tokens=150):
    """
    ROBUST EMAIL SUMMARIZATION with persistent caching.
    """
    # Check cache (using session_state for persistence)
    
    content_hash = get_content_hash(content)
    if content_hash in st.session_state.summary_cache:
        return st.session_state.summary_cache[content_hash]
    
    # Truncate content to prevent token issues
    truncated_content = truncate_email_content(content, max_chars=1500)
    
    # Attempt 1: Full summarization prompt
    prompt = (
        "Summarize this email in 2-3 clear sentences. "
        "Focus on the main point and key information.\n\n"
        f"{truncated_content}\n\nSummary:"
    )
    
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=max_tokens,
            )
        )
        
        summary = safe_extract_gemini_text(response)
        
        if summary and len(summary) > 20:
            # SUCCESS - Cache and return
            st.session_state.summary_cache[content_hash] = summary
            return summary
        
    except Exception as e:
        print(f"Gemini summarization attempt 1 failed: {e}")
    
    # Attempt 2: Simplified prompt (retry)
    try:
        simple_prompt = f"Summarize in 2 sentences:\n{truncated_content[:800]}"
        
        response = gemini_model.generate_content(
            simple_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=100,
            )
        )
        
        summary = safe_extract_gemini_text(response)
        
        if summary and len(summary) > 20:
            st.session_state.summary_cache[content_hash] = summary
            return summary
            
    except Exception as e:
        print(f"Gemini summarization attempt 2 failed: {e}")
    
    # Fallback: Extractive summarization
    fallback_summary = extract_key_sentences(truncated_content, num_sentences=3)
    st.session_state.summary_cache[content_hash] = fallback_summary
    
    return fallback_summary


#CATEGORIZATION 


def categorize_by_rules(subject, body, sender):
    """
    FIXED: Rule-based email categorization with correct security/work classification.
    """
    text = f"{subject} {body} {sender}".lower()
    
    # SECURITY/WORK NOTIFICATIONS 
    # These should NEVER be classified as personal
    security_keywords = [
        'verification code', 'otp', 'one-time password', 'security alert',
        'login attempt', 'unusual activity', 'sign-in', 'password reset',
        'two-factor', '2fa', 'authentication', 'verify your account',
        'confirm your email', 'account security', 'suspicious activity',
        'unauthorized access', 'new device', 'security code', 'verify',
        'verification', 'confirm', 'authentication code', 'access code'
    ]
    
    security_count = sum(1 for kw in security_keywords if kw in text)
    
    if security_count >= 1:
        # Security emails are work-related system notifications
        return 'work'
    
    # 2: SPAM DETECTION 
    spam_keywords = [
        'congratulations! you won', 'claim your prize', 'click here now',
        'limited time offer', 'act now', 'free money', 'make money fast',
        'nigerian prince', 'inheritance', 'lottery winner', 'viagra',
        'weight loss miracle', 'work from home', 'million dollars',
        'verify account immediately', 'suspended account'
    ]
    
    spam_patterns = [
        r'\$\d{1,3}(,\d{3})*(\.\d{2})?',
        r'click here',
        r'act now',
        r'limited time',
    ]
    
    spam_count = sum(1 for kw in spam_keywords if kw in text)
    if spam_count >= 2:
        return 'spam'
    
    for pattern in spam_patterns:
        if len(re.findall(pattern, text, re.IGNORECASE)) >= 2:
            return 'spam'
    
    # 3: FINANCE DETECTION
    finance_keywords = [
        'bank', 'payment', 'invoice', 'receipt', 'transaction', 'credit card',
        'debit', 'charge', 'refund', 'billing', 'subscription', 'paypal',
        'stripe', 'razorpay', 'account balance', 'statement', 'wire transfer',
        'deposit', 'withdrawal', 'tax', 'insurance', 'loan', 'mortgage',
        'financial', 'bill', 'due', 'paid', 'owed', 'upi', 'phonepe',
        'gpay', 'paytm', 'net banking'
    ]
    
    finance_senders = [
        'bank', 'paypal', 'stripe', 'razorpay', 'paytm', 'phonepe',
        'gpay', 'mastercard', 'visa', 'amex', 'banking', 'finance',
        'billing', 'noreply@'
    ]
    
    finance_count = sum(1 for kw in finance_keywords if kw in text)
    sender_match = any(fs in sender.lower() for fs in finance_senders)
    
    if finance_count >= 2 or (finance_count >= 1 and sender_match):
        return 'finance'
    
    # 4: WORK DETECTION
    work_keywords = [
        'meeting', 'project', 'deadline', 'team', 'manager', 'colleague',
        'office', 'conference', 'presentation', 'report', 'client',
        'proposal', 'contract', 'business', 'professional', 'corporate',
        'department', 'employee', 'hr', 'human resources', 'salary',
        'performance review', 'quarterly', 'stakeholder', 'deliverable',
        'milestone', 'sprint', 'agile', 'scrum'
    ]
    
    work_patterns = [
        r'\bQ\d\b',
        r'\d{4}\s*roadmap',
        r'standup',
        r'sync\s+meeting',
        r'all-hands',
    ]
    
    work_count = sum(1 for kw in work_keywords if kw in text)
    has_work_pattern = any(re.search(p, text, re.IGNORECASE) for p in work_patterns)
    
    if work_count >= 3 or (work_count >= 2 and has_work_pattern):
        return 'work'
    
    #5: SUBSCRIPTION/NEWSLETTER 
    subscription_keywords = [
        'unsubscribe', 'newsletter', 'weekly digest', 'daily brief',
        'notification settings', 'email preferences', 'subscription',
        'mailing list'
    ]
    
    sub_count = sum(1 for kw in subscription_keywords if kw in text)
    if sub_count >= 1:
        if work_count >= 1:
            return 'work'
        else:
            return 'personal'
    
    #6: PERSONAL 
    personal_keywords = [
        'hi', 'hey', 'hello', 'dear friend', 'how are you', 'miss you',
        'catch up', 'dinner', 'lunch', 'coffee', 'party', 'birthday',
        'wedding', 'vacation', 'weekend', 'family', 'mom', 'dad',
        'brother', 'sister', 'friend', 'love', 'thanks'
    ]
    
    personal_count = sum(1 for kw in personal_keywords if kw in text)
    personal_domains = ['@gmail.com', '@yahoo.com', '@hotmail.com', '@outlook.com']
    is_personal_domain = any(domain in sender.lower() for domain in personal_domains)
    
    if personal_count >= 2 or (personal_count >= 1 and is_personal_domain):
        return 'personal'
    
    return None


def categorize_email(content):
    """
    FIXED: Email categorization with persistent caching and improved prompt.
    """
    # Check cache (using session_state)
    content_hash = get_content_hash(content)
    if content_hash in st.session_state.category_cache:
        return st.session_state.category_cache[content_hash]
    
    # Extract components
    subject = ""
    body = ""
    sender = ""
    
    for line in content.split('\n'):
        if line.startswith('Subject:'):
            subject = line.replace('Subject:', '').strip()
        elif line.startswith('From:'):
            sender = line.replace('From:', '').strip()
        elif line.startswith('Body:'):
            body = content[content.find('Body:') + 5:].strip()
            break
    
    # Try rule-based categorization first
    rule_category = categorize_by_rules(subject, body, sender)
    
    if rule_category:
        st.session_state.category_cache[content_hash] = rule_category
        return rule_category
    
    # Fallback to Gemini with IMPROVED PROMPT
    truncated_content = truncate_email_content(content, max_chars=1000)
    
    prompt = (
        "Categorize this email into EXACTLY ONE category: work, personal, finance, or spam.\n\n"
        "STRICT RULES:\n"
        "- work: Security alerts, OTPs, verification codes, login warnings, system notifications, "
        "professional emails, meetings, projects, business correspondence\n"
        "- finance: Bank statements, payment receipts, invoices, transactions, credit card alerts, "
        "UPI notifications, billing, financial services\n"
        "- personal: Messages from friends/family, social events, casual conversations, "
        "personal invitations (NOT automated notifications)\n"
        "- spam: Promotional offers, suspicious content, unsolicited marketing, phishing attempts\n\n"
        "IMPORTANT:\n"
        "- Verification codes, OTPs, security alerts = work (NOT personal)\n"
        "- Google/system account notifications = work (NOT personal)\n"
        "- Bank/payment notifications = finance (NOT personal)\n"
        "- Only genuine human correspondence from friends/family = personal\n\n"
        f"{truncated_content}\n\n"
        "Answer with ONLY ONE WORD (work/personal/finance/spam):"
    )
    
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=10,
            )
        )
        
        category = safe_extract_gemini_text(response)
        
        if category:
            category = category.lower().strip()
            valid_categories = ['work', 'personal', 'finance', 'spam']
            
            for valid_cat in valid_categories:
                if valid_cat in category:
                    st.session_state.category_cache[content_hash] = valid_cat
                    return valid_cat
        
    except Exception as e:
        print(f"Gemini categorization failed: {e}")
    
    # Ultimate fallback with improved heuristics
    if any(kw in text.lower() for kw in ['verify', 'otp', 'security', 'login', 'authentication']):
        st.session_state.category_cache[content_hash] = 'work'
        return 'work'
    elif any(kw in subject.lower() for kw in ['payment', 'invoice', 'receipt', 'bank', 'transaction']):
        st.session_state.category_cache[content_hash] = 'finance'
        return 'finance'
    elif any(kw in sender.lower() for kw in ['gmail.com', 'yahoo.com', 'hotmail.com']):
        st.session_state.category_cache[content_hash] = 'personal'
        return 'personal'
    
    st.session_state.category_cache[content_hash] = 'personal'
    return 'personal'



# HELPER FUNCTIONS FOR QUERIES


def summarize_emails_query(query, service):
    """Handle summarization queries with caching."""
    num_emails = extract_number_from_query(query)
    sender_filter = extract_sender_from_query(query)
    
    messages = get_email_messages(service, max_results=num_emails if num_emails else 10)
    
    summaries = []
    for msg in messages[:num_emails if num_emails else 5]:
        details = get_email_message_details(service, msg["id"])
        
        # Apply sender filter if specified
        if sender_filter and sender_filter.lower() not in details["sender"].lower():
            continue
        
        subject = details.get("subject", "No Subject")
        body = details.get("body", "")
        sender = details.get("sender", "Unknown")
        
        email_content = f"Subject: {subject}\nFrom: {sender}\nBody: {body}"
        summary = summarize_email(email_content)
        
        summaries.append({
            "subject": subject,
            "sender": sender,
            "summary": summary
        })
    
    return summaries


def categorize_emails_query(query, service):
    """Handle categorization queries with caching."""
    num_emails = extract_number_from_query(query)
    
    messages = get_email_messages(service, max_results=num_emails if num_emails else 10)
    
    categorized = []
    for msg in messages[:num_emails if num_emails else 5]:
        details = get_email_message_details(service, msg["id"])
        
        subject = details.get("subject", "No Subject")
        body = details.get("body", "")
        sender = details.get("sender", "Unknown")
        
        email_content = f"Subject: {subject}\nFrom: {sender}\nBody: {body}"
        category = categorize_email(email_content)
        
        categorized.append({
            "subject": subject,
            "sender": sender,
            "category": category
        })
    
    return categorized


def summarize_and_categorize_emails(query, service):
    """Handle combined queries with caching."""
    num_emails = extract_number_from_query(query)
    
    messages = get_email_messages(service, max_results=num_emails if num_emails else 10)
    
    results = []
    for msg in messages[:num_emails if num_emails else 5]:
        details = get_email_message_details(service, msg["id"])
        
        subject = details.get("subject", "No Subject")
        body = details.get("body", "")
        sender = details.get("sender", "Unknown")
        
        email_content = f"Subject: {subject}\nFrom: {sender}\nBody: {body}"
        
        # Both operations use caching
        summary = summarize_email(email_content)
        category = categorize_email(email_content)
        
        results.append({
            "subject": subject,
            "sender": sender,
            "summary": summary,
            "category": category
        })
    
    return results



# UTILITY FUNCTIONS


def extract_number_from_query(query):
    """Extract number from queries like 'last 5 emails' or 'recent 10 messages'."""
    numbers = re.findall(r'\b(\d+)\b', query)
    return int(numbers[0]) if numbers else None


def extract_sender_from_query(query):
    """Extract sender name from queries like 'emails from Google' or 'messages from Sarah'."""
    from_match = re.search(r'from\s+([A-Za-z]+)', query, re.IGNORECASE)
    return from_match.group(1) if from_match else None


def detect_query_intent(query):
    """
    Determine user intent: summarize, categorize, both, or regular RAG query.
    
    Returns:
        str: 'summarize', 'categorize', 'both', or 'rag'
    """
    query_lower = query.lower()
    
    has_summarize = any(word in query_lower for word in ['summarize', 'summary', 'summaries', 'tldr'])
    has_categorize = any(word in query_lower for word in ['categorize', 'category', 'categories', 'classify', 'label'])
    
    if has_summarize and has_categorize:
        return 'both'
    elif has_summarize:
        return 'summarize'
    elif has_categorize:
        return 'categorize'
    else:
        return 'rag'


# ORIGINAL RAG FUNCTION 


def answer_query_via_gemini(query, k=10, temperature=0.3, max_tokens=512):
    """
    Retrieve relevant emails from Chroma and generate an answer using Google Gemini.
    """
    try:
        docs = vector_db.similarity_search(query, k=k)
    except Exception:
        retriever = vector_db.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)

    if not docs:
        return "No relevant emails found in memory."

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



# STREAMLIT UI 
st.set_page_config(page_title="AI Email Assistant", page_icon="📧", layout="wide")

if "summary_cache" not in st.session_state:
    st.session_state.summary_cache = {}

if "category_cache" not in st.session_state:
    st.session_state.category_cache = {}


st.sidebar.title("Email Assistant")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Select Mode:",
    ["Chat (RAG)", "Summarize Emails", "Categorize Emails", "Both (Summary + Category)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "**Chat Mode:** Ask questions about your emails\n\n"
    "**Summarize:** Get concise summaries\n\n"
    "**Categorize:** Classify emails into categories\n\n"
    "**Both:** Get summaries with categories"
)

# Cache statistics in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Cache Statistics")
st.sidebar.metric("Summaries Cached", len(st.session_state.summary_cache))
st.sidebar.metric("Categories Cached", len(st.session_state.category_cache))

st.markdown(
    """
    <h1 style='text-align:center;'>AI Email Assistant</h1>
    <p style='text-align:center; font-size:18px;'>
    Your intelligent email companion powered by Gemini + RAG
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# MODE 1: CHAT


if mode == "Chat (RAG)":
    st.subheader("Ask me anything about your emails")
    
    for role, text in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Assistant:** {text}")
    
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input("Your question:", key="chat_input", placeholder="e.g., Any urgent emails?")
    with col2:
        send_button = st.button("Send", use_container_width=True)
    
    if send_button and query.strip():
        st.session_state.chat_history.append(("user", query))
        
        intent = detect_query_intent(query)
        
        if intent == 'summarize':
            summaries = summarize_emails_query(query, service)
            response = "**Email Summaries:**\n\n"
            for i, item in enumerate(summaries, 1):
                response += f"**{i}. {item['subject']}**\n"
                response += f"   *From:* {item['sender']}\n"
                response += f"   *Summary:* {item['summary']}\n\n"
        elif intent == 'categorize':
            categories = categorize_emails_query(query, service)
            response = "**Email Categories:**\n\n"
            for i, item in enumerate(categories, 1):
                response += f"**{i}. {item['subject']}**\n"
                response += f"   *From:* {item['sender']}\n"
                response += f"   *Category:* `{item['category']}`\n\n"
        elif intent == 'both':
            results = summarize_and_categorize_emails(query, service)
            response = "**Email Analysis:**\n\n"
            for i, item in enumerate(results, 1):
                response += f"**{i}. {item['subject']}**\n"
                response += f"   *From:* {item['sender']}\n"
                response += f"   *Category:* `{item['category']}`\n"
                response += f"   *Summary:* {item['summary']}\n\n"
        else:
            response = answer_query_via_gemini(query)
        
        st.session_state.chat_history.append(("assistant", response))
        st.rerun()
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# MODE 2: SUMMARIZE EMAILS


elif mode == "Summarize Emails":
    st.subheader("Email Summarization")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        num_emails = st.slider("Number of emails to summarize:", 1, 20, 5)
    with col2:
        st.write("")
    
    sender_filter = st.text_input("Filter by sender (optional):", placeholder="e.g., Google, Sarah")
    
    if st.button("Generate Summaries", use_container_width=True):
        with st.spinner("Generating summaries..."):
            messages = get_email_messages(service, max_results=num_emails)
            
            summaries = []
            for msg in messages:
                details = get_email_message_details(service, msg["id"])
                
                # Apply filter
                if sender_filter and sender_filter.lower() not in details["sender"].lower():
                    continue
                
                subject = details.get("subject", "No Subject")
                body = details.get("body", "")
                sender = details.get("sender", "Unknown")
                
                email_content = f"Subject: {subject}\nFrom: {sender}\nBody: {body}"
                summary = summarize_email(email_content)
                
                summaries.append({
                    "subject": subject,
                    "sender": sender,
                    "summary": summary
                })
            
            # Display results
            st.success(f"Generated {len(summaries)} summaries")
            
            for i, item in enumerate(summaries, 1):
                with st.expander(f"{i}. {item['subject'][:60]}...", expanded=(i == 1)):
                    st.markdown(f"**From:** {item['sender']}")
                    st.markdown(f"**Subject:** {item['subject']}")
                    st.markdown("**Summary:**")
                    st.info(item['summary'])

# MODE 3: CATEGORIZE EMAILS


elif mode == "Categorize Emails":
    st.subheader("Email Categorization")
    
    num_emails = st.slider("Number of emails to categorize:", 1, 20, 5)
    
    if st.button("Categorize Emails", use_container_width=True):
        with st.spinner("Categorizing emails..."):
            categories = categorize_emails_query(f"categorize last {num_emails} emails", service)
            
            # Display results
            st.success(f"Categorized {len(categories)} emails")
            
            # Summary by category
            category_counts = {}
            for item in categories:
                cat = item['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            st.markdown("###Category Distribution")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Work", category_counts.get('work', 0))
            with col2:
                st.metric("Personal", category_counts.get('personal', 0))
            with col3:
                st.metric("Finance", category_counts.get('finance', 0))
            with col4:
                st.metric("Spam", category_counts.get('spam', 0))
            
            st.markdown("---")
            
            # Detailed results
            for i, item in enumerate(categories, 1):
                # Color-code by category
                category_colors = {
                    'work': '🔵',
                    'personal': '🟢',
                    'finance': '🟡',
                    'spam': '🔴'
                }
                icon = category_colors.get(item['category'], '⚪')
                
                with st.expander(f"{icon} {i}. {item['subject'][:60]}...", expanded=(i == 1)):
                    st.markdown(f"**From:** {item['sender']}")
                    st.markdown(f"**Subject:** {item['subject']}")
                    st.markdown(f"**Category:** `{item['category'].upper()}`")


# MODE 4: BOTH (SUMMARY + CATEGORY)


elif mode == "Both (Summary + Category)":
    st.subheader("Complete Email Analysis")
    
    num_emails = st.slider("Number of emails to analyze:", 1, 20, 5)
    
    if st.button("Analyze Emails", use_container_width=True):
        with st.spinner("Analyzing emails..."):
            results = summarize_and_categorize_emails(f"analyze last {num_emails} emails", service)
            
            # Display results
            st.success(f"Analyzed {len(results)} emails")
            
            # Category distribution
            category_counts = {}
            for item in results:
                cat = item['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            st.markdown("### Category Distribution")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Work", category_counts.get('work', 0))
            with col2:
                st.metric("Personal", category_counts.get('personal', 0))
            with col3:
                st.metric("Finance", category_counts.get('finance', 0))
            with col4:
                st.metric("Spam", category_counts.get('spam', 0))
            
            st.markdown("---")
            
            # Detailed results
            for i, item in enumerate(results, 1):
                category_colors = {
                    'work': '🔵',
                    'personal': '🟢',
                    'finance': '🟡',
                    'spam': '🔴'
                }
                icon = category_colors.get(item['category'], '⚪')
                
                with st.expander(f"{icon} {i}. {item['subject'][:60]}...", expanded=(i == 1)):
                    st.markdown(f"**From:** {item['sender']}")
                    st.markdown(f"**Subject:** {item['subject']}")
                    
                    col_cat, col_sum = st.columns([1, 3])
                    with col_cat:
                        st.markdown(f"**Category:**")
                        st.markdown(f"`{item['category'].upper()}`")
                    with col_sum:
                        st.markdown("**Summary:**")
                        st.info(item['summary'])

