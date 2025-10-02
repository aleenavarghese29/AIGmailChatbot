import marimo as mo

from gmail_api import init_gmail_service, get_email_messages, get_email_message_details

client_file = "client_secret.json"

service = init_gmail_service(client_file)

# Step 2: Fetch the latest 5 messages (change max_results to fetch more)
messages = get_email_messages(service, max_results=5)

for msg in messages:
    details = get_email_message_details(service, msg['id'])
    print("--------------------------------------------------")
    print(f"Subject : {details['subject']}")
    print(f"From    : {details['sender']}")
    print(f"Body    : {details['body'][:500]}...")  # first 500 chars
    print("--------------------------------------------------\n")
