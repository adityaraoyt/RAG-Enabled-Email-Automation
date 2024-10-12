

```markdown
# AI-Powered Email Follow-up Generator

This script generates personalized follow-up emails for MyCompany using the Tiny LLAMA LLM enabled with RAG. It leverages company-specific information and OpenAI's GPT-4 to create tailored, effective follow-up messages.

## Features

- Generates personalized follow-up emails based on the number of days since last contact
- Uses company-specific documents to tailor the email content
- Validates recipient email addresses
- Saves generated emails as Word documents
- Option to send emails directly (requires Gmail account)

## Prerequisites

- Python 3.7+
- Gmail account (for sending emails)

## Installation

1. Clone this repository

2. Set up a local environment using either of these commands
   
   python -m venv myenv
   # or
   virtualenv myenv
   

3. Activate the virtual environment:
   - On Windows:
     ```
     myenv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source myenv/bin/activate
     ```

4. Install the required packages:
   
   pip install -r requirements.txt
   

## Usage

Run the script with the following command-line arguments:


python main.py --days <days> --company <company_name> --email <recipient_email> --sender-email <sender_email> [--send]


Arguments:
- `--days`: Number of days since last contact
- `--company`: Target company name
- `--email`: Recipient's email address
- `--sender-email`: Your email address
- `--send`: (Optional) Flag to send the email immediately

Example:

python -W ignore main.py --days 7 --company comany1 --email john@company1.com --sender-email your@email.com --send


## Directory Structure

- `docs/`: Contains company-specific documents and email templates
  - `<company_name>/`: Subdirectories for each company's documents
  - `email_formats.docx`: Email templates for different follow-up periods
- `responses/`: Generated email responses are saved here as Word documents

## Notes

- Ensure that company-specific documents are placed in the correct subdirectory under `docs/`
- The script uses Gmail's SMTP server for sending emails. Adjust the SMTP settings if using a different email provider
- Make sure to use a secure method to input your email password when sending emails
```
