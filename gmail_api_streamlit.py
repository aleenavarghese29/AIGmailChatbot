import json
from google_apis import create_service

def init_gmail_service_streamlit(gcp_oauth_secrets, api_name='gmail', api_version='v1', scopes=['https://mail.google.com/']):
    """
    Initialize Gmail API service using secrets from Streamlit.
    Returns:
        service: Gmail API service object
    """
    # Create temporary client secret JSON file
    client_file = "client_secret_tmp.json"
    with open(client_file, "w") as f:
        json.dump(gcp_oauth_secrets, f)  # no extra "installed" wrapping

    # Initialize Gmail API service
    service = create_service(client_file, api_name, api_version, scopes)
    return service
