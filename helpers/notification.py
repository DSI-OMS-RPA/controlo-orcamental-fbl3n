import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders

from typing import Tuple, Any, List

import logging
import requests

from helpers.database.postgresql_generic_crud import PostgresqlGenericCRUD
from helpers.configuration import load_json_config

# Set logging for debug.
logging.basicConfig(level=logging.INFO)

# Custom exception class to handle data retrieval errors
class DataRetrievalError(Exception):
    pass

# Custom exception class to handle invalid data format errors
class InvalidDataFormatError(Exception):
    pass

# Fetch SMTP credentials from a specified table and query.
def get_smtp_credentials(crud: PostgresqlGenericCRUD, table: Any, where: Any) -> Tuple[str, str, str, int]:
    """
    Parameters:
    - table: The table to query.
    - where: The query to execute on the table.

    Returns:
    - Tuple[str, str, str, int]: A tuple containing the username, password, server URL, and port for SMTP.
    """

    # Find records with a specific condition in the employee_details_ext table
    where_clause = where['clause']
    where_params = (where['params'],)
    columns = ['username', 'password', 'url', 'port']
    mail_args = crud.read(table, columns, where=where_clause, params=where_params)

    # Check if data was retrieved, raise a custom exception if not
    if not mail_args:
        raise DataRetrievalError("Failed to retrieve mail arguments")

    # Check if the first item in mail_args is a dict, raise a custom exception if not
    if not isinstance(mail_args[0], dict):
        raise InvalidDataFormatError("Invalid data format retrieved")

    try:
        # Attempt to extract SMTP credentials from the first item in mail_args
        username = mail_args[0]['username']
        password = mail_args[0]['password']
        url = mail_args[0]['url']
        port = mail_args[0]['port']
    except KeyError as e:
        # Catch any KeyError exceptions (missing expected keys), and raise a custom exception with a descriptive error message
        raise InvalidDataFormatError(f"Missing expected key: {e}") from e

    # Return the extracted SMTP credentials as a tuple
    return username, password, url, port

# Connects to the SMTP server using credentials fetched from a data source.
def connect_smtp(crud: PostgresqlGenericCRUD, table: Any, where: Any) -> Tuple[smtplib.SMTP, str]:
    """
    Parameters:
    - table: The table to query.
    - where: The query to execute on the table.

    Returns:
    - A tuple containing the SMTP server object and the username.
    """
    try:
        # Get SMTP credentials
        username, password, server, port = get_smtp_credentials(crud, table, where)

    except ValueError as e:
        # Log and re-raise any exceptions encountered while fetching credentials
        logging.error(f"Failed to get SMTP credentials: {e}")
        raise

    try:

        # Connect to the SMTP server
        smtp_server = smtplib.SMTP(server, port)

        # Upgrade the connection to a secure encrypted SSL/TLS connection
        # smtp_server.starttls()
        # Login to the SMTP server
        # smtp_server.login(username, password)

    except smtplib.SMTPException as e:
        # Log and re-raise any exceptions encountered while connecting to the SMTP server
        logging.error(f"Failed to connect to SMTP server: {e}")
        raise

    # Return the SMTP server object and the username
    return smtp_server, username


def attach_image(msg, image_path):
    """
    Attach an image to the email message with a specific Content-ID.
    """
    with open(image_path, 'rb') as img:
        part = MIMEImage(img.read())
        part.add_header('Content-ID', f"<{os.path.basename(image_path)}>")
        part.add_header('Content-Disposition', 'inline', filename=os.path.basename(image_path))
        msg.attach(part)

# Sends an email with optional attachments.
def send_email(crud: PostgresqlGenericCRUD, to: str, subject: str, message_body: str, html_body=False, attachment_paths: List[str] = None, cc: List[str] = None, bcc: List[str] = None, from_address: str = None) -> None:
    # Your function implementation here

    from helpers.utils import is_valid_email, is_image_file

    """
    Parameters:
    - to: The email address of the recipient.
    - subject: The subject of the email.
    - message_body: The body of the email.
    - attachment_paths: A list of file paths to attach to the email.
    - cc: A list of email addresses to send a carbon copy to.
    - bcc: A list of email addresses to send a blind carbon copy to.
    - from_address: The email address to send the email from.
    """

    if attachment_paths is None:
        attachment_paths = []
    if cc is None:
        cc = []
    if bcc is None:
        bcc = []

    # Load configuration settings
    config = load_json_config()

    # database configs
    database = config['database']
    table = database['webmail']['table']
    where = database['webmail']['where']

    # Connect to the SMTP server
    server, username = connect_smtp(crud, table, where)

    if from_address and is_valid_email(from_address):
        # Set the username to the provided from_address
        username = from_address

    try:

        # Create a message object
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = to
        msg['Subject'] = subject

        # Add CC and BCC recipients if provided
        if cc:
            msg['Cc'] = ', '.join(cc)
        if bcc:
            msg['Bcc'] = ', '.join(bcc)

        # Compile all recipient addresses (including CC and BCC)
        all_recipients = [to] + cc + bcc

        if html_body:
            # Attach the HTML message to the email
            msg.attach(MIMEText(message_body, 'html', 'utf-8'))
        else:
            # Attach the plain text message to the email
            msg.attach(MIMEText(message_body, 'plain', 'utf-8'))

        # Attach files if provided
        for attachment_path in attachment_paths:
            if is_image_file(attachment_path):
                # Function to attach the image with a Content-ID
                attach_image(msg, attachment_path)
            else:
                with open(attachment_path, 'rb') as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f"attachment; filename= {attachment_path}")
                msg.attach(part)

        # send result
        send_result = server.sendmail(
            username, all_recipients, msg.as_string())

        # Check if the email was successfully sent
        if send_result == {}:
            logging.info("Email sent successfully.")
            return True
        else:
            logging.error(f"Failed to send email to some or all recipients: {send_result}")
            return False

    except smtplib.SMTPException as e:
        logging.error(f"Error sending email: {str(e)}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        cleanup_connection(server)

# Ensure to cleanup connection
def cleanup_connection(smtp_server: smtplib.SMTP) -> None:
    """
    Closes the connection to the SMTP server.

    Parameters:
    - smtp_server: The SMTP server object.
    """
    if smtp_server:
        # Close the SMTP server connection
        smtp_server.quit()


# Sends an SMS notification
def send_sms(url, uid, pw, o, m, n):

    # Validate user inputs
    if not (uid and pw and o and m and n):
        logging.error("One or more required parameters are missing.")
        return

    # Define the parameters as a dictionary
    params = {
        'UID': uid,
        'PW': pw,
        'O': o,
        'M': m,
        'N': n
    }

    # Make the GET request with the parameters
    response = requests.get(url, params=params)

    # Check the response
    return response.status_code == 200
