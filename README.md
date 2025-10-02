

# AIGmailChatbot

**An intelligent Gmail assistant built with Streamlit, Transformers, and Gmail API.**

It can:

* Classify emails into categories like updates, promotions, social media, spam, and more.
* Summarize email content using advanced generative AI models.
* Detect urgent or spam emails automatically.
* Search emails by sender or keyword.
* Respond to natural language queries like “Show urgent emails” or “Summarize emails.”

---

## Features

* **Generative AI Summarization**: Summarize emails using `facebook/bart-large-cnn`.
* **Email Classification**: Uses `jason23322/email-classifier-distilbert` to categorize emails.
* **Urgency Detection**: Highlights emails containing urgent keywords.
* **Spam Detection**: Identifies potential spam emails.
* **Search Functionality**: Find emails by sender (`from:example@gmail.com`) or keyword (`search:invoice`).
* **User-friendly Interface**: Built with Streamlit for interactive experience.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/aleenavarghese29/AIGmailChatbot.git
cd AIGmailChatbot
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your Gmail API credentials:

* Place your `client_secret.json` in the project root.
* Optionally, create a `.env` file for storing tokens securely.

---

## Usage

Run the Streamlit app:

```bash
streamlit run enhanced_chatbot.py
```

Then type queries like:

* `Show urgent emails`
* `Summarize emails`
* `from:example@gmail.com`
* `search:invoice`

---

## Dependencies

* `streamlit`
* `torch`
* `transformers`
* `google-api-python-client`
* `google-auth-httplib2`
* `google-auth-oauthlib`


---

## Notes

* This project is a **generative AI application** because it uses models to summarize and understand email content.
* Do **not upload your `client_secret.json` or token files** to GitHub. Add them to `.gitignore`.

---

## License

MIT License

---

