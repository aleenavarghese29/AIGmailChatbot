import re
import spacy
import dateparser
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")

def preprocess_email(text):
    # Clean text
    clean = re.sub(r'<.*?>', '', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    # Remove signatures
    signature_patterns = [r"Regards,.*", r"Best,.*", r"Sent from my.*", r"--.*"]
    for pat in signature_patterns:
        clean = re.sub(pat, '', clean, flags=re.IGNORECASE)
    # Remove stopwords
    words = [w for w in clean.split() if w.lower() not in STOPWORDS]
    clean = " ".join(words)
    # Extract dates
    dates = dateparser.search.search_dates(clean) or []
    # Extract entities
    doc = nlp(clean)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return {
        "clean_text": clean,
        "dates": dates,
        "entities": entities
    }
