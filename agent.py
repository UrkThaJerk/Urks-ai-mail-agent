import os
from dotenv import load_dotenv
from openai import OpenAI
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

from video_agent import process_video_jobs

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_gmail_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def process_emails():
    client = get_openai_client()
    service = get_gmail_service()
    results = service.users().messages().list(userId="me", q="is:unread").execute()
    messages = results.get("messages", [])
    if not messages:
        print("No unread messages found.")
        return

    for msg in messages:
        txt = service.users().messages().get(userId="me", id=msg["id"]).execute()
        snippet = txt.get("snippet", "")
        print(f"Processing email snippet: {snippet}")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional assistant. Draft a reply to this email."},
                {"role": "user", "content": snippet}
            ]
        )

        message = response.choices[0].message
        draft_content = message["content"] if isinstance(message, dict) else message.content
        print(f"Generated Draft: {draft_content}")


def run_agent():
    agent_type = os.getenv("URKS_AGENT_TYPE", "mail").strip().lower()
    if agent_type == "video":
        process_video_jobs()
        return
    process_emails()


if __name__ == "__main__":
    run_agent()
