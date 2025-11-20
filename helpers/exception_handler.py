import sys
import traceback
import logging
from datetime import datetime
from typing import Dict, Any

class ExceptionHandler:
    """
    A class that handles exceptions and logs detailed information about them.
    Integrates with EmailSender for robust email notifications.

    Attributes:
        log_file (str): The name of the log file to store the exception information.
        log_level (int): The logging level for the exception.
        log_format (str): The format of the log messages.
    """

    def __init__(self,
                 crud,
                 email_sender,
                 config: Dict[str, Any],
                 log_file: str = 'error_log.log',
                 log_level: int = logging.DEBUG,
                 log_format: str = '%(asctime)s:%(levelname)s:%(message)s'):
        """
        Initialize the ExceptionHandler with custom logging settings and EmailSender.

        Args:
            crud: Database CRUD operations interface
            email_sender (EmailSender): Configured EmailSender instance
            config (Dict[str, Any]): Configuration dictionary containing email settings
            log_file (str): Path to log file
            log_level (int): Logging level
            log_format (str): Format string for log messages
        """
        logging.basicConfig(filename=log_file, level=log_level, format=log_format)

        self.crud = crud
        self.email_sender = email_sender
        self.config = config

    def get_exception(self, exception: Exception, send_email: bool = True) -> Dict[str, Any]:
        """
        Capture, log, and store detailed information about an exception.

        Args:
            exception (Exception): The exception to handle
            send_email (bool): Whether to send email notification

        Returns:
            Dict[str, Any]: Dictionary containing error information
        """
        error_info = {
            "error_message": str(exception),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
            "exception_type": type(exception).__name__,
            "exception_args": exception.args,
            "exception_module": type(exception).__module__,
            "exception_file": exception.__traceback__.tb_frame.f_code.co_filename,
            "exception_line": exception.__traceback__.tb_lineno,
            "status": "pending"
        }

        # Store error information and send email if requested
        if error_info:
            self.store_error(error_info)
            if send_email:
                self.send_error_report(error_info)

        return error_info

    def handle_uncaught_exception(self,
                                exc_type: type,
                                exc_value: Exception,
                                exc_traceback: traceback,
                                send_email: bool = False) -> Dict[str, Any]:
        """
        Handle uncaught exceptions with detailed logging and optional email notification.

        Args:
            exc_type (type): Exception type
            exc_value (Exception): Exception instance
            exc_traceback (traceback): Exception traceback
            send_email (bool): Whether to send email notification

        Returns:
            Dict[str, Any]: Dictionary containing error information
        """
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return {}

        error_info = {
            "error_message": str(exc_value),
            "traceback": ''.join(traceback.format_tb(exc_traceback)),
            "timestamp": datetime.now().isoformat(),
            "exception_type": exc_type.__name__,
            "exception_args": exc_value.args,
            "exception_module": exc_type.__module__,
            "status": "pending"
        }

        logging.error(f"Uncaught Exception: {error_info}")

        # Store and optionally send email notification
        if error_info:
            self.store_error(error_info)
            if send_email:
                self.send_error_report(error_info)

        return error_info

    def store_error(self, error_info: Dict[str, Any]) -> None:
        """
        Store error information in the database.

        Args:
            error_info (Dict[str, Any]): Dictionary containing error details
        """
        try:
            cleaned_info = self.clean_error_info(error_info)

            # Extract values and keys into a list
            values = tuple(cleaned_info.values())
            columns = list(cleaned_info.keys())

            # Store the error information
            success = self.crud.create('error_logs', values, columns)
            if success:
                logging.info("Error information stored in database successfully.")
            else:
                logging.error("Failed to store error information in database.")
        except Exception as e:
            logging.error(f"Exception while storing error information: {e}")

    def send_error_report(self, error_info: Dict[str, Any]) -> bool:
        """
        Send an error report using EmailSender with a template.

        Args:
            error_info (Dict[str, Any]): Information about the error

        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        try:
            # Prepare email configuration
            report_config = {
                'to': self.config.get('to', []),
                'subject': self.config.get('subject', 'Error Report'),
                'from_mail': self.config.get('from_mail'),
                'cc': self.config.get('cc', [])
            }

            # Generate environment info
            environment = self.config.get('environment', 'production')

            # Format error details for better readability
            error_details = (
                f"Exception Type: {error_info['exception_type']}\n"
                f"Message: {error_info['error_message']}\n"
                f"Module: {error_info.get('exception_module', 'N/A')}\n"
                f"File: {error_info.get('exception_file', 'N/A')}\n"
                f"Line: {error_info.get('exception_line', 'N/A')}\n"
                f"Full Traceback:\n{error_info['traceback']}"
            )

            # Send template email using EmailSender
            return self.email_sender.send_template_email(
                report_config=report_config,
                alert_type='danger',
                alert_title='Exception Alert',
                alert_message="An exception occurred in the application.",
                error_details=error_details,
                environment=environment,
                timestamp=error_info['timestamp'],
                action_button={
                    'text': 'View Details',
                    'url': self.config.get('error_dashboard_url', '#')
                }
            )
        except Exception as e:
            logging.error(f"Failed to send error report email: {e}")
            return False

    def clean_error_info(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and standardize error information for storage.

        Args:
            error_info (Dict[str, Any]): Raw error information

        Returns:
            Dict[str, Any]: Cleaned error information
        """
        cleaned_info = {
            "error_message": error_info.get("error_message", "").strip(),
            "traceback": error_info.get("traceback", "").strip(),
            "timestamp": error_info.get("timestamp", datetime.now().isoformat()),
            "exception_type": error_info.get("exception_type", ""),
            "exception_args": str(error_info.get("exception_args", "")),
            "exception_module": error_info.get("exception_module", ""),
            "exception_file": error_info.get("exception_file", ""),
            "exception_line": error_info.get("exception_line", 0),
            "status": error_info.get("status", "pending")
        }
        return cleaned_info
