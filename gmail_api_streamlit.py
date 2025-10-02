import json
from google_apis import create_service

def init_gmail_service_streamlit(gcp_oauth_secrets, api_name='gmail', api_version='v1', scopes=['https://mail.google.com/']):
    """
    Initialize Gmail API in Streamlit Cloud using secrets.
    
    Args:
        gcp_oauth_secrets (dict): Dictionary from st.secrets["gcp_oauth"]
        api_name (str): API name, default 'gmail'
        api_version (str): API version, default 'v1'
        scopes (list): Gmail API scopes, default full access

    Returns:
        service: Gmail API service object
    """
    # Create temporary client secret JSON file
    client_file = "client_secret_tmp.json"
    with open(client_file, "w") as f:
        json.dump({"installed": gcp_oauth_secrets}, f)

    # Initialize Gmail API service
    service = create_service(client_file, api_name, api_version, scopes)
    return service
