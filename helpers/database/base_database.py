from abc import ABC, abstractmethod

class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors."""
    pass


class BaseDatabase(ABC):
    """
    Abstract base class for database operations.

    Attributes:
        config (dict): Configuration parameters for the database connection.
        connection: The database connection object.
    """

    def __init__(self, config):
        self.config = config
        self.connection = None

    @abstractmethod
    def connect(self):
        """Establish the database connection."""
        pass

    @abstractmethod
    def disconnect(self):
        """Close the database connection."""
        pass

    @abstractmethod
    def execute_query(self, query, params=None):
        """
        Execute a database query.

        Args:
            query (str or dict): The query to be executed.
            params (tuple or dict, optional): Parameters for the query.

        Returns:
            Result of the query execution.
        """
        pass
