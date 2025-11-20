import re
import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from typing import Any, List, Dict, Optional, Union, Generator
from jinja2 import Environment, FileSystemLoader
from contextlib import contextmanager
from pathlib import Path
from functools import lru_cache

# Set logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailError(Exception):
    """Base exception class for email-related errors."""
    pass

class DataRetrievalError(EmailError):
    """Exception class to handle data retrieval errors."""
    pass

class InvalidDataFormatError(EmailError):
    """Exception class to handle invalid data format errors."""
    pass

class SMTPConnectionError(EmailError):
    """Exception class to handle SMTP connection errors."""
    pass

class EmailSender:
    """A class to handle email sending operations with various features like templates and attachments.

    The EmailSender class provides a robust interface for sending both regular and template-based emails
    with support for attachments, HTML content, and customizable alert templates.

    Attributes:
        smtp_configs (Dict[str, str]): SMTP server configuration containing server, port, and optional credentials
        IMAGE_EXTENSIONS (set): Supported image file extensions
        EMAIL_REGEX (Pattern): Regular expression for email validation
        ALERT_COLORS (Dict[str, str]): Color mapping for different alert types
        REQUIRED_SMTP_PARAMS (set): Required SMTP configuration parameters
    """

    # Class constants
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    ALERT_COLORS = {
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'info': '#17a2b8'
    }
    REQUIRED_SMTP_PARAMS = {'server', 'port'}

    def __init__(self, smtp_configs: Dict[str, str]):
        """
        Initialize EmailSender with SMTP configurations.

        Args:
            smtp_configs (Dict[str, str]): SMTP server configuration dictionary containing:
                - server (str): SMTP server address
                - port (str): SMTP server port
                - username (str, optional): SMTP username
                - password (str, optional): SMTP password
        """

        self._smtp_configs = smtp_configs
        self._validate_config()

        # Initialize the template environment during object creation
        self._template_env = self._create_jinja2_environment()

    @property
    def smtp_configs(self) -> Dict[str, str]:
        """Get the current SMTP configuration."""
        return self._smtp_configs.copy()

    def update_smtp_config(self, new_config: Dict[str, str]) -> None:
        """
        Update SMTP configuration with new settings.

        Args:
            new_config (Dict[str, str]): New SMTP configuration to apply

        Raises:
            DataRetrievalError: If required parameters are missing
            InvalidDataFormatError: If parameters are in invalid format
        """
        # Create a new config combining current and new settings
        updated_config = self._smtp_configs.copy()
        updated_config.update(new_config)

        # Validate before applying
        self._smtp_configs = updated_config
        try:
            self._validate_config()
        except Exception as e:
            # Rollback on validation failure
            self._smtp_configs = self._smtp_configs.copy()
            raise e

    def _validate_config(self) -> None:
        """
        Validate SMTP configuration parameters.

        Validates both the presence of required parameters and their format.
        Checks port is numeric and server is not empty.

        Raises:
            DataRetrievalError: If required parameters are missing
            InvalidDataFormatError: If parameters are in invalid format
        """
        # Check for missing parameters
        missing_params = self.REQUIRED_SMTP_PARAMS - set(self.smtp_configs.keys())
        if missing_params:
            raise DataRetrievalError(f"Missing required parameters: {missing_params}")

        # Validate port is numeric
        try:
            port = int(self.smtp_configs['port'])
            if port <= 0:
                raise InvalidDataFormatError("Port must be a positive number")
        except ValueError:
            raise InvalidDataFormatError("Port must be a valid number")

        # Validate server is not empty
        if not self.smtp_configs['server'].strip():
            raise InvalidDataFormatError("Server address cannot be empty")

    @contextmanager
    def _smtp_connection(self) -> Generator[smtplib.SMTP, None, None]:
        """
        Context manager for SMTP connections.

        Yields:
            smtplib.SMTP: The SMTP connection object.

        Raises:
            SMTPConnectionError: If connection fails.
        """
        smtp_server = None
        try:
            smtp_server = self._connect_smtp()
            yield smtp_server
        finally:
            self._cleanup_connection(smtp_server)

    @staticmethod
    @lru_cache(maxsize=100)
    def is_valid_email(email: str) -> bool:
        """
        Validate an email address using regex with caching.

        Args:
            email (str): Email address to validate

        Returns:
            bool: True if email is valid, False otherwise
        """
        return bool(EmailSender.EMAIL_REGEX.match(email))

    @staticmethod
    def is_image_file(filepath: str) -> bool:
        """
        Check if a file is an image based on its extension.

        Args:
            filepath (str): Path to the file

        Returns:
            bool: True if file is an image, False otherwise
        """
        return Path(filepath).suffix.lower() in EmailSender.IMAGE_EXTENSIONS

    def _connect_smtp(self) -> smtplib.SMTP:
        """
        Connect to the SMTP server using credentials.

        Returns:
            smtplib.SMTP: Connected SMTP server object

        Raises:
            SMTPConnectionError: If connection fails
        """
        try:
            server = self.smtp_configs['server']
            port = int(self.smtp_configs['port'])
            username = self.smtp_configs.get('username')
            password = self.smtp_configs.get('password')

            if username and password:
                smtp_server = smtplib.SMTP_SSL(server, port)
                smtp_server.login(username, password)
            else:
                smtp_server = smtplib.SMTP(server, port)

            smtp_server.ehlo()
            return smtp_server
        except (smtplib.SMTPException, ValueError) as e:
            raise SMTPConnectionError(f"Failed to connect to SMTP server: {e}")

    def _attach_file(self, msg: MIMEMultipart, file_path: str) -> None:
        """
        Attach a file to the email message.

        Args:
            msg (MIMEMultipart): Email message object
            file_path (str): Path to file to attach
        """
        if not Path(file_path).exists():
            logger.warning(f"Attachment not found: {file_path}")
            return

        try:
            if self.is_image_file(file_path):
                with open(file_path, 'rb') as img:
                    part = MIMEImage(img.read())
                    part.add_header('Content-ID', f"<{Path(file_path).name}>")
                    part.add_header('Content-Disposition', 'inline', filename=Path(file_path).name)
            else:
                with open(file_path, 'rb') as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', f"attachment; filename= {Path(file_path).name}")

            msg.attach(part)
        except IOError as e:
            logger.error(f"Failed to attach file {file_path}: {e}")

    @staticmethod
    def get_rgba_color(hex_color: str, opacity: float = 1.0) -> str:
        """
        Convert hex color to rgba color with opacity.

        Args:
            hex_color (str): Hex color code
            opacity (float): Opacity value between 0 and 1

        Returns:
            str: RGBA color string
        """
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {opacity})"

    @property
    def template_env(self) -> Environment:
        """Get the Jinja2 template environment."""
        if self._template_env is None:
            self._template_env = self._create_jinja2_environment()
        return self._template_env

    def _create_jinja2_environment(self) -> Environment:
        """
        Create and configure Jinja2 environment with custom filters.

        Returns:
            Environment: Configured Jinja2 environment
        """
        template_dir = Path(__file__).parent.parent
        env = Environment(loader=FileSystemLoader(template_dir))

        # Add custom filters
        def format_date(value: str, fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
            """Convert string date to formatted string."""
            if isinstance(value, str):
                try:
                    date_obj = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                    return date_obj.strftime(fmt)
                except ValueError:
                    return value
            return value

        def default_date(value: Optional[str] = None, fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
            """Return current date if value is None."""
            if value is None:
                return datetime.now().strftime(fmt)
            return value

        env.filters['date'] = format_date
        env.filters['default_date'] = default_date  # Added missing filter
        env.globals['now'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return env

    def _validate_alert_type(self, alert_type: str) -> None:
        """
        Validate alert type against supported types.

        Args:
            alert_type (str): Type of alert to validate

        Raises:
            InvalidDataFormatError: If alert type is not supported
        """
        if alert_type not in self.ALERT_COLORS:
            valid_types = ', '.join(self.ALERT_COLORS.keys())
            raise InvalidDataFormatError(
                f"Invalid alert type: {alert_type}. Must be one of: {valid_types}"
            )

    def generate_alert(
        self,
        alert_type: str,
        alert_title: str,
        alert_message: str,
        file_names: Optional[List[str]] = None,
        alert_link: Optional[str] = None,
        table_data: Optional[List[Dict[str, Any]]] = None,
        company_logo: Optional[str] = None,
        summary_data: Optional[List[Dict[str, Any]]] = None,
        table_summary: Optional[List[str]] = None,
        total_records: Optional[int] = None,
        show_pagination: bool = False,
        file_status: Optional[Dict[str, str]] = None,
        file_metadata: Optional[Dict[str, str]] = None,
        error_details: Optional[str] = None,
        action_button: Optional[Dict[str, str]] = None,
        environment: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> str:
        """
        Generate an HTML alert message using a Jinja template.

        Args:
            alert_type (str): Type of alert ('success', 'warning', 'danger', 'info')
            alert_title (str): Title of the alert
            alert_message (str): Main message content
            file_names (Optional[List[str]]): List of processed file names
            alert_link (Optional[str]): URL for detail button
            table_data (Optional[List[Dict[str, Any]]]): Table data as list of dictionaries
            company_logo (Optional[str]): Path to company logo
            summary_data (Optional[List[Dict[str, Any]]]): Summary statistics
            table_summary (Optional[List[str]]): Summary row for table
            total_records (Optional[int]): Total number of records
            show_pagination (bool): Whether to show pagination info
            file_status (Optional[Dict[str, str]]): Status for each file
            file_metadata (Optional[Dict[str, str]]): Additional metadata for each file
            error_details (Optional[str]): Error message details
            action_button (Optional[Dict[str, str]]): Action button config
            environment (Optional[str]): Environment name
            timestamp (Optional[str]): Timestamp for the alert

        Returns:
            str: Rendered HTML template
        """
        self._validate_alert_type(alert_type)
        alert_color = self.ALERT_COLORS[alert_type]
        template = self.template_env.get_template('./template/alert_template.html')

        table_headers = list(table_data[0].keys()) if table_data else None

        return template.render(
            html_title='Alert Notification',
            alert_type=alert_type,
            alert_title=alert_title,
            alert_message=alert_message,
            file_names=file_names,
            alert_link=alert_link,
            alert_color=alert_color,
            table_headers=table_headers,
            table_data=table_data,
            company_logo=company_logo,
            summary_data=summary_data,
            table_summary=table_summary,
            total_records=total_records,
            show_pagination=show_pagination,
            file_status=file_status,
            file_metadata=file_metadata,
            error_details=error_details,
            action_button=action_button,
            environment=environment,
            timestamp=timestamp
        )

    def send_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        message_body: str,
        html_body: bool = False,
        attachment_paths: Optional[List[str]] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        from_address: Optional[str] = None
    ) -> bool:
        """
        Send an email with optional attachments.

        Args:
            to (Union[str, List[str]]): Recipient email address(es)
            subject (str): Email subject
            message_body (str): Email body content
            html_body (bool): Whether the message body is HTML
            attachment_paths (Optional[List[str]]): List of file paths to attach
            cc (Optional[List[str]]): CC recipients
            bcc (Optional[List[str]]): BCC recipients
            from_address (Optional[str]): Sender email address

        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        attachment_paths = attachment_paths or []
        cc = cc or []
        bcc = bcc or []

        try:
            with self._smtp_connection() as server:
                msg = MIMEMultipart()
                username = from_address if from_address and self.is_valid_email(from_address) else self.smtp_configs.get('username')

                msg['From'] = username
                msg['To'] = to if isinstance(to, str) else ', '.join(to)
                msg['Subject'] = subject

                if cc:
                    msg['Cc'] = ', '.join(cc)
                if bcc:
                    msg['Bcc'] = ', '.join(bcc)

                all_recipients = (to if isinstance(to, list) else [to]) + cc + bcc
                msg.attach(MIMEText(message_body, 'html' if html_body else 'plain', 'utf-8'))

                for attachment_path in attachment_paths:
                    self._attach_file(msg, attachment_path)

                server.sendmail(username, all_recipients, msg.as_string())
                logger.info("Email sent successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False

    def send_template_email(
        self,
        report_config: Dict[str, Union[str, List[str]]],
        alert_type: str,
        alert_title: str,
        alert_message: str,
        attachment_paths: Optional[List[str]] = None,
        file_names: Optional[List[str]] = None,
        alert_link: Optional[str] = None,
        table_data: Optional[List[Dict[str, Any]]] = None,
        company_logo: Optional[str] = None,
        summary_data: Optional[List[Dict[str, Any]]] = None,
        table_summary: Optional[List[str]] = None,
        total_records: Optional[int] = None,
        show_pagination: bool = False,
        file_status: Optional[Dict[str, str]] = None,
        file_metadata: Optional[Dict[str, str]] = None,
        error_details: Optional[str] = None,
        action_button: Optional[Dict[str, str]] = None,
        environment: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> bool:
        """
        Send an email notification with a template.

        Args:
            report_config (Dict[str, Union[str, List[str]]]): Email report configuration
            alert_type (str): Type of alert
            alert_title (str): Title of the alert
            alert_message (str): Main message content
            attachment_paths (Optional[List[str]]): List of file paths to attach
            file_names (Optional[List[str]]): List of processed file names
            alert_link (Optional[str]): URL for detail button
            table_data (Optional[List[Dict[str, Any]]]): Table data
            company_logo (Optional[str]): Path to company logo
            summary_data (Optional[List[Dict[str, Any]]]): Summary statistics
            table_summary (Optional[List[str]]): Summary row for table
            total_records (Optional[int]): Total number of records
            show_pagination (bool): Whether to show pagination info
            file_status (Optional[Dict[str, str]]): Status for each file
            file_metadata (Optional[Dict[str, str]]): Additional metadata for each file
            error_details (Optional[str]): Error message details
            action_button (Optional[Dict[str, str]]): Action button config
            environment (Optional[str]): Environment name
            timestamp (Optional[str]): Timestamp for the alert

        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        try:
            message_body = self.generate_alert(
                alert_type=alert_type,
                alert_title=alert_title,
                alert_message=alert_message,
                file_names=file_names,
                alert_link=alert_link,
                table_data=table_data,
                company_logo=company_logo,
                summary_data=summary_data,
                table_summary=table_summary,
                total_records=total_records,
                show_pagination=show_pagination,
                file_status=file_status,
                file_metadata=file_metadata,
                error_details=error_details,
                action_button=action_button,
                environment=environment,
                timestamp=timestamp
            )

            logger.info(f"Sending template email notification to {report_config['to']}")

            return self.send_email(
                report_config['to'],
                report_config['subject'],
                message_body,
                html_body=True,
                cc=report_config.get('cc'),
                from_address=report_config.get('from_mail'),
                attachment_paths=attachment_paths
            )

        except Exception as e:
            logger.error(f"Failed to send template email: {str(e)}")
            return False

    @staticmethod
    def _cleanup_connection(smtp_server: Optional[smtplib.SMTP]) -> None:
        """
        Close the connection to the SMTP server.

        Args:
            smtp_server (Optional[smtplib.SMTP]): SMTP server to clean up
        """
        if smtp_server:
            try:
                smtp_server.quit()
            except smtplib.SMTPException as e:
                logger.error(f"Error closing SMTP connection: {str(e)}")
            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")
