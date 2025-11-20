import logging
import os
import sys
import io
from datetime import datetime
from logging.handlers import RotatingFileHandler


class LoggerManager:
    """
    LoggerManager sets up a logging system with UTF-8 support for Windows environments.
    Handles file and console output with proper encoding configuration.
    Prevents duplicate handlers that can cause repeated log messages.

    Attributes:
        log_dir (str): Directory where log files will be saved.
        log_level (int): Logging level.
        log_filename (str): Generated log filename based on the current datetime.
        logger (logging.Logger): The logger instance for the class.
    """

    def __init__(self, log_dir='logs', log_level=logging.DEBUG):
        """
        Initializes the LoggerManager with the specified log directory and log level.
        Ensures only one instance of handlers is added to prevent log duplication.

        Args:
            log_dir (str): Directory to save log files.
            log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        """
        self.log_dir = log_dir
        self.log_level = log_level
        self.log_filename = self.generate_log_filename()
        self.logger = None

        # Force UTF-8 encoding on Windows for stderr (logging output)
        self._configure_utf8_for_windows()

        # Setup logging (only once)
        self.setup_logging()

    @staticmethod
    def _configure_utf8_for_windows():
        """
        Force UTF-8 encoding on Windows console to prevent Unicode errors in logging.
        This resolves 'charmap' codec errors when logging contains Unicode characters.
        """
        if sys.platform == 'win32':
            try:
                # Reconfigure stderr to use UTF-8 encoding
                if hasattr(sys.stderr, 'reconfigure'):
                    sys.stderr.reconfigure(encoding='utf-8')
                else:
                    # Fallback for older Python versions
                    sys.stderr = io.TextIOWrapper(
                        sys.stderr.buffer,
                        encoding='utf-8',
                        errors='replace'
                    )

                # Also configure stdout for consistency
                if hasattr(sys.stdout, 'reconfigure'):
                    sys.stdout.reconfigure(encoding='utf-8')
                else:
                    sys.stdout = io.TextIOWrapper(
                        sys.stdout.buffer,
                        encoding='utf-8',
                        errors='replace'
                    )
            except Exception:
                # Silent failure - logging not yet initialized
                pass

    def generate_log_filename(self):
        """
        Generates a log filename based on the current datetime.

        Returns:
            str: The generated log filename with full path.
        """
        return datetime.now().strftime(f'{self.log_dir}/agt003dsi_%Y%m%d%H%M%S.log')

    def setup_logging(self):
        """
        Sets up the logging configuration with file and console handlers.
        Ensures the log directory exists and configures UTF-8 encoding.

        CRITICAL: Removes existing handlers from root logger to prevent duplication.
        """
        # Ensure the log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Get root logger
        root_logger = logging.getLogger()

        # IMPORTANT: Remove all existing handlers to prevent duplication
        # This is critical when LoggerManager is instantiated multiple times
        for handler in root_logger.handlers[:]:  # Use slice to avoid modification during iteration
            root_logger.removeHandler(handler)
            handler.close()

        # Set root logger level
        root_logger.setLevel(self.log_level)

        # Create a file handler with UTF-8 encoding
        file_handler = logging.FileHandler(
            self.log_filename,
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        # Create a console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        # Add handlers to root logger (only once)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        # Assign the class logger
        self.logger = logging.getLogger(__name__)

    def get_logger(self, name=None):
        """
        Retrieves a logger by name or returns the default logger.

        Args:
            name (str): Optional name of the logger to retrieve.

        Returns:
            logging.Logger: The retrieved logger.
        """
        return logging.getLogger(name) if name else self.logger

    def add_console_handler(self):
        """
        Adds a console handler to the root logger with UTF-8 encoding.
        DEPRECATED: Handlers are automatically added in setup_logging().
        """
        root_logger = logging.getLogger()
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.add_handler(console_handler, '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def add_rotating_file_handler(self, max_bytes=10485760, backup_count=5):
        """
        Adds a rotating file handler to the root logger with UTF-8 encoding.

        Args:
            max_bytes (int): Maximum file size in bytes before rotating.
            backup_count (int): Number of backup files to keep.
        """
        rotating_handler = RotatingFileHandler(
            self.log_filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        self.add_handler(
            rotating_handler,
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def add_handler(self, handler, format_str):
        """
        Adds a specified handler to the root logger with the given format.

        Args:
            handler (logging.Handler): The handler to add.
            format_str (str): The format string for the log messages.
        """
        handler.setLevel(self.log_level)
        handler.setFormatter(logging.Formatter(
            format_str,
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logging.getLogger().addHandler(handler)

    def get_log_filename(self):
        """
        Retrieves the generated log filename.

        Returns:
            str: The generated log filename.
        """
        return self.log_filename
