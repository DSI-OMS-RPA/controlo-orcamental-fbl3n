from typing import Any, Dict, List, Optional, Tuple
import logging
from datetime import date, datetime

logger = logging.getLogger(__name__)


class PostgresqlGenericCRUD:
    """Generic CRUD operations for any table."""

    def __init__(self, db_client):
        """
        Initialize the PostgresqlGenericCRUD class.

        Args:
            db_client: An instance of a database client (e.g., PostgreSQLClient).
        """
        self.db_client = db_client

    def _get_table_columns(self, table: str, show_id: bool = False) -> List[str]:
        """
        Get the column names of a table, optionally including the 'id' column.

        Args:
            table (str): The table name.
            show_id (bool): If True, include the 'id' column. Default is False.

        Returns:
            list: List of column names.
        """
        query = """
        SELECT COLUMN_NAME
        FROM information_schema.COLUMNS
        WHERE TABLE_NAME = %s
        AND is_identity = 'NO'
        """
        if not show_id:
            query += "AND column_name != 'id' "
        query += "ORDER BY ordinal_position"

        try:
            result = self.db_client.execute_query(query, (table,), fetch_as_dict=True)
            columns = [row['column_name'] for row in result]
            return columns
        except Exception as e:
            logger.error(f"Failed to get table columns. Error: {e}")
            raise

    def _format_dates(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format date fields in a record to a readable string format.

        Args:
            record (dict): The record with potential date fields.

        Returns:
            dict: The record with formatted date fields.
        """
        for key, value in record.items():
            if isinstance(value, (date, datetime)):
                record[key] = value.strftime(
                    '%Y-%m-%d %H:%M:%S') if isinstance(value, datetime) else value.strftime('%Y-%m-%d')
        return record

    def create(self, table: str, values: List[Tuple[Any]], columns: List[str] = None) -> None:
        """
        Create new records in the specified table.

        Args:
            table (str): The table name.
            values (list of tuples): List of tuples of values to insert.
            columns (list, optional): List of column names. If None, columns will be inferred from the table schema.
        """
        if columns is None:
            columns = self._get_table_columns(table)

        # Ensure values is a list of tuples
        if not isinstance(values, list):
            values = [values]
        elif not all(isinstance(v, tuple) for v in values):
            raise ValueError("Values must be a tuple or a list of tuples.")

        for value_tuple in values:
            if len(value_tuple) != len(columns):
                raise ValueError(f"Number of values {len(value_tuple)} does not match number of columns {len(columns)}")

        columns_str = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))
        query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"

        try:
            self.db_client.execute_batch_query(query, values)
            logger.info("Records inserted.")
            return True
        except Exception as e:
            logger.error(f"Failed to insert records. Error: {e}")
            raise

    def read(self, table: str, columns: List[str] = None, where: str = "", params: Tuple[Any] = None, show_id: bool = False) -> List[Dict[str, Any]]:
        """
        Read records from the specified table.

        Args:
            table (str): The table name.
            columns (list, optional): List of column names to retrieve. If None, all columns will be retrieved.
            where (str, optional): WHERE clause for filtering records.
            params (tuple, optional): Tuple of parameters for the WHERE clause.
            show_id (bool, optional): If True, include the 'id' column. Default is False.

        Returns:
            list: List of records as dictionaries.
        """
        if columns is None:
            columns = self._get_table_columns(table, show_id=show_id)

        columns_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {columns_str} FROM {table}"
        if where:
            query += f" WHERE {where}"
        try:
            result = self.db_client.execute_query(
                query, params, fetch_as_dict=False)
            records = [self._format_dates(
                dict(zip(columns, row))) for row in result]
            logger.info("Records found.")
            return records
        except Exception as e:
            logger.error(f"Failed to read records. Error: {e}")
            raise

    def update(self, table: str, updates: Dict[str, Any], where: str, params: Tuple[Any]) -> None:
        """
        Update records in the specified table.

        Args:
            table (str): The table name.
            updates (dict): Dictionary of columns and their new values.
            where (str): WHERE clause for identifying records to update.
            params (tuple): Tuple of parameters for the WHERE clause.
        """
        set_clause = ", ".join([f"{col} = %s" for col in updates.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {where}"
        values = tuple(updates.values()) + params
        try:
            self.db_client.execute_query(query, values)
            logger.info("Records updated.")
            return True
        except Exception as e:
            logger.error(f"Failed to update records. Error: {e}")
            raise

    def delete(self, table: str, where: str = "", params: Tuple[Any] = None) -> None:
        """
        Delete records from the specified table.

        Args:
            table (str): The table name.
            where (str, optional): WHERE clause for identifying records to delete. If empty, all records will be deleted.
            params (tuple, optional): Tuple of parameters for the WHERE clause.
        """
        query = f"DELETE FROM {table}"
        if where:
            query += f" WHERE {where}"
        try:
            self.db_client.execute_query(query, params)
            logger.info("Records deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete records. Error: {e}")
            raise

    def execute_raw_query(self, query: str, params: Optional[Any] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a raw SQL query.

        Args:
            query (str): The SQL query to execute.
            params (Any, optional): Parameters to bind to the query. Default is None.

        Returns:
            Optional[List[Dict[str, Any]]]: If the query is a SELECT query, returns a list of dictionaries
                                        representing rows. Otherwise, returns None.
        """
        try:
            is_select_query = query.strip().lower().startswith('select')
            if is_select_query:
                # Execute the SELECT query and get results as dictionaries
                result = self.db_client.execute_query(query, params, fetch_as_dict=True)
                # Format dates in each record
                formatted_result = [self._format_dates(record) for record in result]
                return formatted_result
            else:
                # Execute the non-SELECT query
                self.db_client.execute_query(query, params)
                logger.info("Query executed successfully.")
                return None
        except Exception as e:
            logger.error(f"Failed to execute raw query. Error: {e}")
            raise
