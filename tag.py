import os.path
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import base64
from email.mime.text import MIMEText
import json
from openai import OpenAI
import os
import tiktoken
import re
from datetime import datetime, timedelta
import time
from bs4 import BeautifulSoup

# Load secrets from environment variables (Railway/production) or secrets.json (local)
def load_secrets():
    # Try environment variable first (for Railway/production)
    openai_key = os.environ.get('OPENAI_API_KEY')

    if openai_key:
        return {'openai_api_key': openai_key}

    # Fallback to secrets.json for local development
    try:
        with open('secrets.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Neither OPENAI_API_KEY environment variable nor secrets.json file found.")
    except json.JSONDecodeError:
        raise ValueError("secrets.json is not properly formatted JSON.")

# Initialize OpenAI client with API key from secrets
secrets = load_secrets()
client = OpenAI(api_key=secrets['openai_api_key'])

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.modify']

# Constants
CLASSIFICATION_LABELS = ["ATTN", "FK-U", "MARKETING", "TAKE-A-LOOK", "HMMMM", "ATTN-COFOUNDER"]

# Maximum character limit for email content
MAX_CONTENT_LENGTH = 500  # Aggressive truncation

def load_prompt_template():
    """Loads the prompt template from prompt.txt"""
    try:
        with open('prompt.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError("prompt.txt file not found. Please create it with your classification instructions.")

def classify_email(sender, subject, content, email_date):
    """Classifies email using GPT-4"""
    prompt_template = load_prompt_template()
    prompt = prompt_template.format(
        sender=sender,
        subject=subject,
        content=content,
        email_date=email_date
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[
                {"role": "system", "content": "You are an email classifier that categorizes emails. Always respond in valid JSON format with 'classification' and 'reason' fields."},
                {"role": "user", "content": prompt}
            ]
        )
        # Parse the response text as JSON
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        return {"classification": "ERROR", "reason": f"Failed to parse GPT response as JSON: {str(e)}"}
    except Exception as e:
        return {"classification": "ERROR", "reason": f"Failed to classify: {str(e)}"}

def get_gmail_service():
    """Gets or creates Gmail API service."""
    creds = None
    
    # The file token.pickle stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)

def count_tokens(text, model="gpt-4"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def get_email_body(payload):
    """Extracts email body from Gmail API payload, preferring plain text over HTML."""
    plain_text = None
    html_text = None

    def extract_from_parts(parts):
        """Recursively extract text from MIME parts."""
        nonlocal plain_text, html_text

        for part in parts:
            mime_type = part.get('mimeType', '')

            # If this part has subparts, recurse
            if 'parts' in part:
                extract_from_parts(part['parts'])

            # Check if this is a text part
            elif mime_type == 'text/plain':
                data = part['body'].get('data', '')
                if data:
                    plain_text = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')

            elif mime_type == 'text/html':
                data = part['body'].get('data', '')
                if data:
                    html_text = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')

    # Check if message has parts (multipart)
    if 'parts' in payload:
        extract_from_parts(payload['parts'])
    else:
        # Single part message
        mime_type = payload.get('mimeType', '')
        data = payload['body'].get('data', '')

        if data:
            decoded = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            if mime_type == 'text/plain':
                plain_text = decoded
            elif mime_type == 'text/html':
                html_text = decoded
            else:
                # Unknown type, try to decode anyway
                plain_text = decoded

    # Prefer plain text, fall back to HTML
    if plain_text:
        return clean_email_content(plain_text, is_html=False)
    elif html_text:
        return clean_email_content(html_text, is_html=True)
    else:
        return 'No content'

def extract_text_from_html(html):
    """Converts HTML to clean plain text using BeautifulSoup."""
    soup = BeautifulSoup(html, 'html.parser')

    # Remove script, style, and other non-content tags
    for tag in soup(['script', 'style', 'meta', 'link', 'noscript', 'head', 'header', 'footer', 'nav', 'aside']):
        tag.decompose()

    # Get text
    text = soup.get_text(separator=' ')
    return text


def clean_email_content(text, is_html=False):
    """EXTREME cleaning - maximizes token savings by being ultra-aggressive."""

    # If HTML, convert to text first
    if is_html or '<html' in text.lower() or '<body' in text.lower():
        text = extract_text_from_html(text)

    # Remove ALL URLs (more aggressive pattern)
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'www\.[^\s]+', '', text)
    text = re.sub(r'\S+\.(com|org|net|io|co|ai|dev)[^\s]*', '', text, flags=re.IGNORECASE)

    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # Remove phone numbers
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)

    # Remove all invisible characters
    text = re.sub(r'[\u200B-\u200D\uFEFF\u00A0\u2060\u180E\u00AD\u034F\u061C\u115F\u1160\u17B4\u17B5\u2800\u3164]', '', text)

    # Remove standalone punctuation and brackets
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'\{\s*\}', '', text)

    # Aggressively remove common marketing phrases and patterns
    marketing_patterns = [
        r'View in browser.*',
        r'View this email.*',
        r'View online.*',
        r'Read online.*',
        r'Open in browser.*',
        r'View image:.*',
        r'Follow image link:.*',
        r'Caption:.*',
        r'Click here.*',
        r'Learn more.*',
        r'Read more.*',
        r'Shop now.*',
        r'Buy now.*',
        r'Order now.*',
        r'Subscribe.*',
        r'Unsubscribe.*',
        r'Update your preferences.*',
        r'Manage your subscription.*',
        r'Copyright ©.*',
        r'All rights reserved.*',
        r'You are receiving this.*',
        r'This email was sent.*',
        r'Our mailing address.*',
        r'No longer want to receive.*',
        r'Too many emails\?.*',
        r'Forward to a friend.*',
        r'Share on.*',
        r'Follow us on.*',
        r'Connect with us.*',
        r'Terms and conditions.*',
        r'Privacy policy.*',
        r'Terms of service.*',
        r'Help center.*',
        r'Customer support.*',
        r'Contact us.*',
        r'Add to calendar.*',
        r'Download our app.*',
        r'Get the app.*',
        r'Available on.*',
        r'App Store.*',
        r'Google Play.*',
        r'Sent from my.*',
        r'Get Outlook for.*',
        r'To view this.*',
        r'If you cannot see.*',
    ]

    for pattern in marketing_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # Remove quoted replies (lines starting with >)
    text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)

    # Remove email signatures (anything after common signature markers)
    text = re.sub(r'\n--\s*\n.*', '', text, flags=re.DOTALL)
    text = re.sub(r'\nBest regards,.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\nThanks,.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\nSincerely,.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\nCheers,.*', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove lines that are just repeated characters (decorative separators)
    text = re.sub(r'^(.)\1{3,}$', '', text, flags=re.MULTILINE)

    # Remove lines with just symbols/punctuation
    text = re.sub(r'^[\*\-\=\#\~\|\>\<\+\_\•\◦\▪\▫\→\←\↑\↓\/\\\[\]\{\}\(\)]+\s*$', '', text, flags=re.MULTILINE)

    # Remove social media handles/links
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)

    # Remove "Presented by" and sponsor sections
    text = re.sub(r'Presented by.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Sponsored by.*?\n', '', text, flags=re.IGNORECASE)

    # Clean up whitespace aggressively
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\n\s*\n+', '\n', text)  # All multiple newlines to single
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Trim lines

    # Remove very short lines (likely fragments/formatting)
    lines = text.split('\n')
    meaningful_lines = []

    for line in lines:
        stripped = line.strip()
        # Only keep lines with meaningful content
        # Must have at least 10 characters and contain actual words
        if len(stripped) >= 10 and sum(c.isalpha() for c in stripped) >= 5:
            meaningful_lines.append(stripped)

    text = ' '.join(meaningful_lines)  # Join with spaces instead of newlines

    # Final aggressive truncation - only keep first N characters
    if len(text) > MAX_CONTENT_LENGTH:
        # Try to cut at a sentence boundary
        truncated = text[:MAX_CONTENT_LENGTH]
        last_period = truncated.rfind('.')
        last_space = truncated.rfind(' ')

        if last_period > MAX_CONTENT_LENGTH * 0.7:  # If period is reasonably close to end
            text = truncated[:last_period + 1]
        elif last_space > MAX_CONTENT_LENGTH * 0.8:  # Otherwise cut at word boundary
            text = truncated[:last_space]
        else:
            text = truncated

        text += '...'

    # Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def create_labels_if_needed(service):
    """Creates our classification labels if they don't exist."""
    results = service.users().labels().list(userId='me').execute()
    existing_labels = {label['name']: label['id'] for label in results.get('labels', [])}
    
    # Create any missing labels
    for label_name in CLASSIFICATION_LABELS:
        if label_name not in existing_labels:
            label_body = {
                'name': label_name,
                'messageListVisibility': 'show',
                'labelListVisibility': 'labelShow'
            }
            service.users().labels().create(userId='me', body=label_body).execute()

def update_email_label(service, message_id, classification):
    """Updates the label of an email based on its classification, ensuring only one label at a time."""
    try:
        # Get current labels for the message
        msg = service.users().messages().get(userId='me', id=message_id).execute()
        
        # Get label name mappings
        label_results = service.users().labels().list(userId='me').execute()
        
        # Remove existing classification labels
        classification_label_ids = [
            label['id'] for label in label_results.get('labels', [])
            if label['name'] in CLASSIFICATION_LABELS
        ]
        
        if classification_label_ids:
            service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'removeLabelIds': classification_label_ids}
            ).execute()
        
        # Add the new label
        new_label_id = next(
            (label['id'] for label in label_results.get('labels', [])
            if label['name'] == classification),
            None
        )
        
        if new_label_id:
            service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'addLabelIds': [new_label_id]}
            ).execute()
        else:
            print(f"Warning: Could not find label ID for classification '{classification}'")
            fallback_label_id = next(
                (label['id'] for label in label_results.get('labels', [])
                if label['name'] == "HMMMM"),
                None
            )
            if fallback_label_id:
                service.users().messages().modify(
                    userId='me',
                    id=message_id,
                    body={'addLabelIds': [fallback_label_id]}
                ).execute()
            else:
                print("Error: Could not find HMMMM label")
        
    except Exception as e:
        print(f"Error in update_email_label: {str(e)}")
        raise

def read_emails_in_date_range():
    """Reads and processes emails from Nov 1 2023 to present."""
    service = get_gmail_service()
    
    # Create labels if they don't exist1
    create_labels_if_needed(service)
    
    start_date = (datetime.now() - timedelta(days=20)).strftime('%Y/%m/%d')
    query = f'in:inbox -in:spam after:{start_date}'
    
    try:
        results = service.users().messages().list(userId='me', q=query).execute()
        messages = results.get('messages', [])
        
        while 'nextPageToken' in results:
            time.sleep(2)
            results = service.users().messages().list(
                userId='me',
                q=query,
                pageToken=results['nextPageToken']
            ).execute()
            messages.extend(results.get('messages', []))
            print(f"Fetched {len(messages)} messages so far...")

        if not messages:
            print('No messages found.')
            return
        
        print(f'Processing {len(messages)} emails:')
        print('-' * 50)
        
        for i, message in enumerate(messages):
            if i > 0 and i % 5 == 0:
                print(f"Processed {i} messages. Pausing briefly...")
                time.sleep(3)
            
            # Get full message details including labels
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            
            # Get current labels
            current_labels = set(msg.get('labelIds', []))
            
            # Get label name to ID mapping
            results = service.users().labels().list(userId='me').execute()
            label_map = {label['name']: label['id'] for label in results.get('labels', [])}
            
            # Check if message has any of our classification labels already
            should_skip = any(
                label_map.get(label) in current_labels 
                for label in CLASSIFICATION_LABELS
            )
            
            if should_skip:
                print(f"Skipping message - already has a classification label")
                continue
            
            # Process headers and content
            headers = msg['payload']['headers']
            subject = next((header['value'] for header in headers if header['name'].lower() == 'subject'), 'No Subject')
            sender = next((header['value'] for header in headers if header['name'].lower() == 'from'), 'No Sender')
            email_date = next((header['value'] for header in headers if header['name'].lower() == 'date'), 'No Date')

            # Skip emails from support@luma.com
            if 'support@luma.com' in sender.lower():
                print(f"Skipping email from support@luma.com")
                continue

            # Extract email body using extreme cleaning
            text = get_email_body(msg['payload'])

            token_count = count_tokens(text)
            char_count = len(text)
            print(f'Token count: {token_count} | Chars: {char_count}')
            
            classification = classify_email(sender, subject, text, email_date)
            
            if classification['classification'] not in ['ERROR']:
                update_email_label(service, message['id'], classification['classification'])
            
            print(f'From: {sender}')
            print(f'Subject: {subject}')
            print('Content:')
            print(text)
            print('\nClassification:')
            print(f"Type: {classification['classification']}")
            print(f"Reason: {classification['reason']}")
            print('-' * 50)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def test_specific_email():
    """Test function to process and tag emails only from ddolin@a16z.com"""
    service = get_gmail_service()
    
    # Create labels if they don't exist
    create_labels_if_needed(service)
    
    # Specific query for emails from ddolin@a16z.com
    query = 'from:ddolin@a16z.com'
    
    try:
        results = service.users().messages().list(userId='me', q=query).execute()
        messages = results.get('messages', [])
        
        if not messages:
            print('No messages found from ddolin@a16z.com')
            return
        
        print(f'Processing {len(messages)} emails from ddolin@a16z.com:')
        print('-' * 50)
        
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            
            headers = msg['payload']['headers']
            subject = next((header['value'] for header in headers if header['name'].lower() == 'subject'), 'No Subject')
            sender = next((header['value'] for header in headers if header['name'].lower() == 'from'), 'No Sender')
            email_date = next((header['value'] for header in headers if header['name'].lower() == 'date'), 'No Date')

            # Extract email body using extreme cleaning
            text = get_email_body(msg['payload'])

            # Get classification from GPT
            classification = classify_email(sender, subject, text, email_date)

            print(f'Date: {email_date}')
            print(f'Subject: {subject}')
            print('Content:')
            print(text)
            print('\nClassification:')
            print(f"Type: {classification['classification']}")
            print(f"Reason: {classification['reason']}")
            
            # Apply the label
            if classification['classification'] not in ['ERROR']:
                update_email_label(service, message['id'], classification['classification'])
                print(f"Applied label: {classification['classification']}")
            
            print('-' * 50)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    # test_specific_email()
    read_emails_in_date_range()
