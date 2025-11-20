from pathlib import Path
import random
import chardet
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
from functools import wraps
import subprocess
import pandas as pd
import win32api
import logging
import psutil
import time
import re
import os

from helpers.logger_manager import LoggerManager

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up and returns a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Initialize the logger manager
    logger_manager = LoggerManager()

    # Get the logger with the specified name
    logger = logger_manager.get_logger(name)
    return logger

def timed(func):
    """
    Decorator for logging the execution time of a function.

    This decorator measures the time it takes for the decorated function to run and logs the duration.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: A wrapped version of the original function that logs execution time.

    Example:
        @timed
        def process_data():
            # Function logic
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Execute the function
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        logging.info(f"{func.__name__} took {elapsed_time:.2f} seconds")  # Log execution time
        return result  # Return the result of the function
    return wrapper

def retry(max_retries=3, delay=1, backoff=2, max_delay=None, jitter=0.5, exceptions=(Exception,), logger=None, on_failure=None):
    """
    Decorator for retrying a function with exponential backoff and optional jitter.

    Args:
        max_retries (int): Maximum number of retry attempts. Default is 3.
        delay (int or float): Initial delay between retries in seconds. Default is 1.
        backoff (int or float): Factor by which the delay is multiplied after each retry. Default is 2.
        max_delay (int or float): Maximum delay between retries in seconds. If None, no limit. Default is None.
        jitter (int or float): Random jitter added to delay (to avoid retry synchronization). Default is 0.5.
        exceptions (tuple): Tuple of exception classes to catch and retry on. Default is (Exception,).
        logger (logging.Logger): Logger instance for logging. Default is None, which will use the root logger.
        on_failure (callable): Optional callback function executed after final retry failure. Default is None.

    Returns:
        function: A wrapped version of the original function with retry logic.

    Raises:
        Exception: Reraise the last exception encountered if max_retries is exceeded.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            log = logger or logging  # Use provided logger or root logger

            while attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    log.error(f"Attempt {attempt} failed for {func.__name__} with args={args}, kwargs={kwargs}. Error: {e}")

                    if attempt >= max_retries:
                        log.error(f"Max retries exceeded for function {func.__name__}")
                        if on_failure:
                            on_failure(e, *args, **kwargs)  # Call failure handler
                        raise

                    # Add random jitter to avoid synchronized retries
                    jitter_value = random.uniform(0, jitter)
                    sleep_time = current_delay + jitter_value

                    # Cap the sleep time if max_delay is specified
                    if max_delay:
                        sleep_time = min(sleep_time, max_delay)

                    log.info(f"Retrying in {sleep_time:.2f} seconds (attempt {attempt}/{max_retries})...")
                    time.sleep(sleep_time)
                    current_delay *= backoff  # Exponentially increase the delay

        return wrapper
    return decorator

def run_application(software):
    """
    Search for and run an application on the system.
    This function searches for the specified software on all available drives and runs it if found.
    It also measures the time taken to find and run the software.

    Args:
        software (str): The name of the software to search for and run.

    Returns:
        bool: True if the software is found and run successfully, False otherwise.
    """

    logger = setup_logger(__name__)

    # Function to search a drive for the software
    def search_drive(drive, software):
        for root, dirs, files in os.walk(drive):
            if software in files:
                return os.path.join(root, software)
        return None

    # Function to check if the process is running
    def is_process_running(process_name):
        for proc in psutil.process_iter(['pid', 'name']):
            if process_name.lower() in proc.info['name'].lower():
                return True
        return False

    # Check if the software is already running
    if is_process_running(software):
        logger.info(f"{software} is already running.")
        return True

    # Measure the start time
    start_time = time.time()

    # Get all available drives
    drives = win32api.GetLogicalDriveStrings().split('\000')[:-1]

    # Search each drive sequentially
    for drive in drives:
        software_path = search_drive(drive, software)
        if software_path:
            # Measure the end time
            end_time = time.time()
            elapsed_time = end_time - start_time

            logger.info(f"{software} found at {software_path}. Starting {software}...")
            subprocess.Popen(software_path)
            logger.info(f"Time taken to find and open {software}: {elapsed_time:.2f} seconds")
            return True

    # Measure the end time if not found
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.warning(f"{software} not found.")
    logger.info(f"Time taken to search for {software}: {elapsed_time:.2f} seconds")
    return False

def find_by_field_value(data_list, field, value_to_find):
    """
    Searches for a dictionary in a list of dictionaries where a specified field has a specific value.
    If no dictionary with the specified value is found, it returns the dictionary where the field value is None.

    Parameters:
    data_list (list): List of dictionaries to search through.
    field (str): The field name to search in each dictionary.
    value_to_find (any): The value to search for in the specified field.

    Returns:
    dict: The dictionary with the specified field value, or the one with the field value as None.
    """
    # Define the fallback dictionary with the field value as None
    fallback = next((item for item in data_list if item[field] is None), None)

    # Search for the dictionary with the given field value
    result = next((item for item in data_list if item[field] == value_to_find), fallback)

    return result

def generate_template(template, variables):
    """
    Generate a string by substituting variables into a template.

    Args:
        template (str): The template string with placeholders for variables.
        variables (dict): A dictionary containing the variable names and their values.

    Returns:
        str: The generated string with variables substituted.
    """
    return template.format(**variables)

def convert_date(date_str):
    """
    Convert a date string from the format 'dd/mm/yyyy' to Portuguese format 'dd de Month'.

    Args:
    date_str (str): The date string in the format 'dd/mm/yyyy'.

    Returns:
    str: The date string in the format 'dd de Month' in Portuguese.
    """
    # Convert string to datetime object
    date_obj = datetime.strptime(date_str, "%d/%m/%Y")

    # Dictionary for translating months to Portuguese
    months_in_portuguese = {
        "January": "Janeiro", "February": "Fevereiro", "March": "Março",
        "April": "Abril", "May": "Maio", "June": "Junho",
        "July": "Julho", "August": "Agosto", "September": "Setembro",
        "October": "Outubro", "November": "Novembro", "December": "Dezembro"
    }

    # Formatting the date to 'dd de Month' format
    formatted_date = date_obj.strftime("%d de %B")

    # Extracting the month in English
    month_english = date_obj.strftime("%B")

    # Replacing the English month with the Portuguese equivalent
    return formatted_date.replace(month_english, months_in_portuguese[month_english])


def is_valid_email(email):
    """
    Validate an email address.

    Args:
        email (str): The email address to validate.

    Returns:
        bool: True if the email address is valid, False otherwise.
    """
    # Regex for validating an email address
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)


def is_image_file(filepath):
    """
    Check if a file is an image based on its extension.

    Parameters:
    - filepath: The path of the file to check.

    Returns:
    - True if the file is an image, False otherwise.
    """
    # Define a set of common image file extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}

    # Extract the file extension and check if it is in the set
    _, ext = os.path.splitext(filepath)
    return ext.lower() in image_extensions


def json_to_html(json_data):
    """Converts JSON data to an HTML table.

    Args:
        json_data: A dictionary containing the JSON data or None.

    Returns:
        str: The HTML representation of the JSON data as a table.
    """

    if json_data is None:
        return "<html><body><h1>Error Report</h1><p>No data available to display.</p></body></html>"

    html = "<html><body><h1>Error Report</h1><table border='1'>"
    for key, value in json_data.items():
        html += f"<tr><th>{key}</th><td>{value}</td></tr>"
    html += "</table></body></html>"
    return html


async def get_ad_user(identity: str = None, email: str = None):
    """
    Retrieve and parse Active Directory user information.

    This function executes a PowerShell command to retrieve properties of an Active Directory user
    specified by their identity. The output is then parsed to extract key-value pairs from the data.

    The PowerShell output is decoded from bytes to a string using various encodings. The parsed data is returned as a dictionary of properties.

    Args:
    identity (str): The identity of the Active Directory user.

    Returns:
    dict: A dictionary containing the parsed user properties.

    Raises:
    Exception: If the PowerShell command execution fails or returns an error.
    """
    # Format the PowerShell command with the provided identity
    if identity:
        command = f"Get-ADUser -Identity {identity} -Properties *"
    elif email:
        command = f"Get-ADUser -Filter {{Emailaddress -eq '{email}'}} -Properties *"
    else:
        raise Exception("No identity or email provided.")

    # Execute the PowerShell command
    result = subprocess.run(
        ["powershell", "-Command", command], capture_output=True)

    if result.returncode != 0:
        error_message = result.stderr.decode('utf-8', errors='replace')
        raise Exception(f"Error: {error_message}")

    # List of possible encodings
    encodings = ['utf-8', 'windows-1252', 'iso-8859-1', 'cp850']

    # Try decoding with different encodings
    for encoding in encodings:
        try:
            output = result.stdout.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise Exception("Failed to decode output with known encodings.")

    # Split the output into lines and parse it into a dictionary
    lines = output.strip().split('\n')
    parsed_data = {}
    for line in lines:
        parts = line.split(':', 1)
        if len(parts) == 2:
            key = parts[0].strip()
            value = parts[1].strip()
            parsed_data[key] = value

    return parsed_data

# Function to generate the HTML alert
def generate_alert(alert_type, alert_title, alert_message, data_list=None, alert_link=None):
    """
    Generate an HTML alert message using a Jinja template.

    Args:
        alert_type (str): The type of alert (e.g., 'warning', 'danger', 'info', 'success').
        alert_title (str): The title of the alert message.
        alert_message (str): The main content of the alert message.
        data_list (list, optional): A list of dictionaries to display as additional data.
        alert_link (str, optional): A link to include in the alert message.

    Returns:
        str: An HTML string representing the alert message.
    """

    # Set the color based on the type of alert
    if alert_type == 'success':
        alert_color = '#28a745'  # Green to success
    elif alert_type == 'warning':
        alert_color = '#ffc107'  # Yellow for warning
    elif alert_type == 'danger':
        alert_color = '#dc3545'  # Red for error
    else:
        alert_color = '#333333'  # Standard color if the type is unknown

    # Load the Jinja template for the alert
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('alert_template.html')

    # Render the template with the provided data
    html_output = template.render(
        title='Alerta de Processos ETL',
        alert_type=alert_type,
        alert_title=alert_title,
        alert_message=alert_message,
        data_list=data_list,
        alert_link=alert_link,
        alert_color=alert_color
    )

    return html_output

def clean_numeric_column(col):
    """Enhanced clean numeric values with Portuguese number format support"""
    from helpers.number_converter import convert_portuguese_to_english_number

    # Handle missing values
    if pd.isna(col):
        return None

    # Use the enhanced converter that handles Portuguese formats
    converted = convert_portuguese_to_english_number(col)

    # If conversion didn't change the value and it's still a string,
    # return it as-is (might not be a number)
    if converted == col and isinstance(col, str):
        return col.strip() if col.strip() else None

    return converted

def clean_date_string(date_str):
    """Convert date string to SQL Server compatible format"""
    if not date_str or date_str == '':
        return None

    try:
        # Handle the SAP format (dd.mm.yyyy)
        parts = date_str.split('.')
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            day, month, year = parts
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        return date_str
    except:
        return date_str

def detect_and_convert_date_columns(df):
    """
    Dynamically detect columns that might contain dates and convert them to SQL Server format.

    Args:
        df (DataFrame): DataFrame to process

    Returns:
        DataFrame: DataFrame with converted date columns
    """
    # Date-related keywords in column names
    date_keywords = ['data', 'date', 'dt', 'dia', 'entrado', 'entrada', 'lancamento', 'lçto']

    # Function to clean date strings
    def clean_date_string(date_str):
        if not date_str or pd.isna(date_str) or date_str == '':
            return ''

        try:
            # Handle the SAP format (dd.mm.yyyy)
            if isinstance(date_str, str) and '.' in date_str:
                parts = date_str.split('.')
                if len(parts) == 3 and all(part.isdigit() for part in parts):
                    day, month, year = parts
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

            return date_str
        except:
            return date_str

    # Function to check if a column could contain dates
    def is_potential_date_column(col_name, sample_values):
        col_lower = col_name.lower()

        # Check if column name contains date-related keywords
        if any(keyword in col_lower for keyword in date_keywords):
            return True

        # Check for date patterns in sample values
        date_patterns = [
            r'\d{1,2}\.\d{1,2}\.\d{2,4}',  # dd.mm.yyyy
            r'\d{1,2}/\d{1,2}/\d{2,4}',    # dd/mm/yyyy
            r'\d{4}-\d{1,2}-\d{1,2}'       # yyyy-mm-dd
        ]

        for value in sample_values:
            if not isinstance(value, str):
                continue

            if any(re.search(pattern, value) for pattern in date_patterns):
                return True

        return False

    # Process each column
    detected_date_columns = []

    for col in df.columns:
        # Skip columns with numeric data
        if df[col].dtype.kind in 'ifc':  # integer, float, complex
            continue

        # Get sample values (non-empty)
        sample_values = df[col].dropna().astype(str).head(10).tolist()

        if not sample_values:
            continue

        # Check if this could be a date column
        if is_potential_date_column(col, sample_values):
            detected_date_columns.append(col)

            # First replace any NaN values
            df[col] = df[col].fillna('')

            # Then convert non-empty strings to SQL Server format
            df[col] = df[col].apply(lambda x: clean_date_string(x))

            print(f"Detected and converted date column: {col}")

    return df

def detect_file_encoding(file_path: Path, logger: logging.Logger = None,
                        sample_size: int = 8192, min_confidence: float = 0.6) -> str:
    """
    Detect file encoding with optimized sampling and fallback strategy.

    Args:
        file_path: Path to the file
        logger: Optional logger instance
        sample_size: Bytes to sample for detection (default 8KB)
        min_confidence: Minimum confidence threshold for chardet

    Returns:
        str: Detected encoding name

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file cannot be read
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Quick check for empty files
    if file_path.stat().st_size == 0:
        if logger:
            logger.info("Empty file detected, defaulting to utf-8")
        return 'utf-8'

    # Primary detection with chardet using optimized sampling
    try:
        with open(file_path, 'rb') as file:
            # Read from beginning, middle, and end for better detection
            file_size = file_path.stat().st_size
            sample_data = file.read(min(sample_size, file_size))

            # For larger files, also sample from middle
            if file_size > sample_size * 2:
                file.seek(file_size // 2)
                sample_data += file.read(min(sample_size // 2, file_size - file.tell()))

        result = chardet.detect(sample_data)

        if result and result['encoding'] and result['confidence'] >= min_confidence:
            encoding = result['encoding'].lower()
            if logger:
                logger.info(f"Detected encoding: {encoding} (confidence: {result['confidence']:.2f})")
            return encoding

    except (OSError, PermissionError) as e:
        raise PermissionError(f"Cannot read file {file_path}: {e}")
    except Exception as e:
        if logger:
            logger.warning(f"Chardet detection failed: {e}")

    # Fallback with common encodings (ordered by likelihood)
    fallback_encodings = ['utf-8', 'utf-16', 'cp1252', 'iso-8859-1', 'latin1']

    for encoding in fallback_encodings:
        try:
            with open(file_path, 'r', encoding=encoding, errors='strict') as file:
                # Test read a reasonable chunk
                file.read(min(2048, file_path.stat().st_size))
                if logger:
                    logger.info(f"Fallback successful with encoding: {encoding}")
                return encoding

        except UnicodeDecodeError:
            continue
        except Exception as e:
            if logger:
                logger.warning(f"Unexpected error testing {encoding}: {e}")
            continue

    # Last resort - latin1 can decode any byte sequence
    if logger:
        logger.warning("All encoding detection failed, using latin1 as last resort")
    return 'latin1'
