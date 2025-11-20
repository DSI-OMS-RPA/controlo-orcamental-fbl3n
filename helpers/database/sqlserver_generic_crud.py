import re
from typing import Any, Dict, List, Optional, Tuple
import logging
from datetime import date, datetime
import unicodedata
from tqdm import tqdm

import pandas as pd
from helpers.utils import retry

logger = logging.getLogger(__name__)

class SQLServerGenericCRUD:
    """Generic CRUD operations for any table in SQL Server."""

    def __init__(self, db_client):
        """
        Initialize the SQLServerGenericCRUD class.

        Args:
            db_client: An instance of a database client (e.g., SQLServerClient).
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
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = ?"""
        if not show_id:
            query += " AND COLUMN_NAME != 'ID'"
        query += " ORDER BY ORDINAL_POSITION"

        try:
            result = self.db_client.execute_query(query, (table,), fetch_as_dict=True)
            columns = [row['COLUMN_NAME'] for row in result]
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
                record[key] = value.strftime('%Y-%m-%d %H:%M:%S') if isinstance(value, datetime) else value.strftime('%Y-%m-%d')
        return record

    def _infer_column_types(self, values: List[Tuple[Any]], columns: List[str], primary_key: str = None) -> Dict[str, str]:
        """
        Dynamically infer SQL Server column types with smart fallbacks and enhanced date detection.

        Args:
            values (list of tuples): Sample data for type inference.
            columns (list): Column names.
            primary_key (str, optional): Primary key column name.

        Returns:
            dict: Mapping of columns to SQL Server data types.
        """
        import re
        from datetime import datetime, date

        # Define base type mapping
        type_mapping = {
            int: "BIGINT",  # Use BIGINT instead of INT for safety
            float: "FLOAT",  # Use DECIMAL with generous precision/scale
            str: "NVARCHAR(MAX)",
            date: "DATE",
            datetime: "DATETIME",
            bool: "BIT",
            bytes: "VARBINARY(MAX)"
        }

        # Column name hints for date detection
        date_column_patterns = [
            'data', 'date', 'dt', 'dia', 'fecha', 'datum', 'tarih',     # Common date words
            'criado', 'created', 'modified', 'updated', 'timestamp',    # Creation/modification terms
            'nascimento', 'birth', 'inicio', 'start', 'end', 'fim',     # Life cycle terms
            'entrada', 'emissao', 'recepcao', 'receção',                # Process date terms
            'validade', 'expira', 'expiry', 'due'                       # Expiration terms
        ]

        # Common date formats to try
        date_formats = [
            '%d.%m.%Y',      # 31.12.2023 (SAP format)
            '%Y-%m-%d',      # 2023-12-31 (ISO)
            '%d/%m/%Y',      # 31/12/2023
            '%m/%d/%Y',      # 12/31/2023
            '%d-%m-%Y',      # 31-12-2023
            '%Y/%m/%d',      # 2023/12/31
            '%d %b %Y',      # 31 Dec 2023
            '%d %B %Y',      # 31 December 2023
            '%b %d, %Y',     # Dec 31, 2023
            '%B %d, %Y',     # December 31, 2023
            '%d.%m.%y',      # 31.12.23
            '%Y%m%d',        # 20231231
        ]

        inferred_types = {}

        for idx, column in enumerate(columns):
            # Default to NVARCHAR(MAX) for maximum compatibility
            sql_type = "NVARCHAR(MAX)"

            # Check if column name suggests it's a date
            column_lower = column.lower()
            column_suggests_date = any(pattern in column_lower for pattern in date_column_patterns)

            # Get non-None sample values for this column
            sample_values = [row[idx] for row in values if row[idx] is not None and row[idx] != '']

            # If we have some values, try to infer a type
            if sample_values:
                try:
                    # First check if it's already a date/datetime object
                    first_value = sample_values[0]
                    if isinstance(first_value, (date, datetime)):
                        sql_type = "DATETIME"

                    # Then check if all values are of the same type
                    else:
                        first_type = type(first_value)
                        all_same_type = all(isinstance(val, first_type) for val in sample_values)

                        if all_same_type:
                            # Special handling for strings - try to detect dates
                            if first_type == str and (column_suggests_date or len(sample_values) >= 2):
                                # Try to detect date strings
                                date_detected = False

                                # Check if strings match date patterns
                                for sample in sample_values[:5]:  # Check first 5 samples
                                    if not isinstance(sample, str):
                                        continue

                                    sample = sample.strip()

                                    # Skip empty strings
                                    if not sample:
                                        continue

                                    # Common date pattern checks
                                    # Check for MM/DD/YYYY or DD/MM/YYYY or YYYY/MM/DD pattern
                                    if re.match(r'^(\d{1,4})[/.-](\d{1,2})[/.-](\d{1,4})$', sample):
                                        date_detected = True
                                        break

                                    # Check for "DD Month YYYY" or "Month DD, YYYY" pattern
                                    if re.match(r'^(\d{1,2}\s+[A-Za-z]+\s+\d{2,4})|([A-Za-z]+\s+\d{1,2},?\s+\d{2,4})$', sample):
                                        date_detected = True
                                        break

                                    # Try parsing with known date formats
                                    for fmt in date_formats:
                                        try:
                                            datetime.strptime(sample, fmt)
                                            date_detected = True
                                            break
                                        except ValueError:
                                            continue

                                    if date_detected:
                                        break

                                # If we detected a date pattern, use DATETIME
                                if date_detected:
                                    sql_type = "DATETIME"
                                else:
                                    # Use the mapped SQL type if not a date
                                    sql_type = type_mapping.get(first_type, "NVARCHAR(MAX)")
                            else:
                                # Use the mapped SQL type for non-string types
                                sql_type = type_mapping.get(first_type, "NVARCHAR(MAX)")

                            # Additional checks for numeric types
                            if first_type == int:
                                # Check if any value exceeds INT range
                                if any(val > 2147483647 or val < -2147483648 for val in sample_values):
                                    sql_type = "BIGINT"

                            elif first_type == float:
                                # Always use decimal for floats to be safe
                                sql_type = "FLOAT"
                except Exception:
                    # If any error occurs during inference, use NVARCHAR(MAX)
                    sql_type = "NVARCHAR(MAX)"

            # Add primary key constraint if applicable
            if column == primary_key:
                if sql_type == "NVARCHAR(MAX)":
                    sql_type = "NVARCHAR(255) PRIMARY KEY"
                else:
                    sql_type += " PRIMARY KEY"

            inferred_types[column] = sql_type

        return inferred_types

    def prepare_data_for_sql_insertion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for SQL insertion with proper date handling.
        IMPORTANT: This assumes Portuguese numbers have ALREADY been converted by process_sap_dataframe_numbers()

        Args:
            df: DataFrame to prepare

        Returns:
            DataFrame with properly formatted data
        """
        processed_df = df.copy()

        # Get column types
        try:
            table_schema_query = f"""
            SELECT
                COLUMN_NAME,
                DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{self.table_name}'
            """
            schema_result = self.db_client.execute_query(table_schema_query, fetch_as_dict=True)
            column_types = {row['COLUMN_NAME']: row['DATA_TYPE'] for row in schema_result}
        except Exception:
            # If we can't get schema, infer types
            sample_values = [tuple(row) for row in processed_df.head().values]
            column_types = self._infer_column_types(sample_values, list(processed_df.columns))

        # Date formats to try when converting strings to dates
        date_formats = [
            '%d.%m.%Y',      # 31.12.2023 (SAP format)
            '%Y-%m-%d',      # 2023-12-31 (ISO)
            '%d/%m/%Y',      # 31/12/2023
            '%m/%d/%Y',      # 12/31/2023
            '%d-%m-%Y',      # 31-12-2023
            '%Y/%m/%d',      # 2023/12/31
            '%d %b %Y',      # 31 Dec 2023
            '%d %B %Y',      # 31 December 2023
            '%b %d, %Y',     # Dec 31, 2023
            '%B %d, %Y',     # December 31, 2023
        ]

        # Process each column based on its SQL type
        for col in processed_df.columns:
            # Determine the normalized column name (in case of case sensitivity issues)
            norm_col = col.lower().replace(' ', '').replace('.', '')

            # Get the SQL type for this column (from schema or inferred)
            sql_type = None
            for schema_col, col_type in column_types.items():
                if schema_col.lower() == norm_col:
                    sql_type = col_type
                    break

            if not sql_type:
                continue

            # Handle the column based on its SQL type
            if sql_type.upper() in ('DATETIME', 'DATETIME', 'DATE', 'DATETIMEOFFSET'):
                # Process date columns
                try:
                    # First try pandas to_datetime conversion which is often successful
                    processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')

                    # For any values that failed conversion, try manual parsing
                    mask = processed_df[col].isna() & df[col].notna()

                    if mask.any():
                        # Copy the original values for date parsing
                        original_values = df.loc[mask, col].astype(str).copy()

                        # Try each date format
                        for date_format in date_formats:
                            still_na = processed_df[col].isna() & df[col].notna()
                            if not still_na.any():
                                break

                            for idx in original_values.index:
                                if idx in still_na and still_na[idx]:
                                    date_str = original_values[idx]
                                    try:
                                        date_val = datetime.strptime(date_str, date_format)
                                        processed_df.at[idx, col] = date_val
                                    except ValueError:
                                        continue

                    # Format dates in SQL Server format
                    # Convert to strings in SQL Server format for any valid dates
                    valid_dates = ~processed_df[col].isna()
                    if valid_dates.any():
                        # Format as 'YYYY-MM-DD HH:MM:SS.000'
                        processed_df.loc[valid_dates, col] = processed_df.loc[valid_dates, col].dt.strftime('%Y-%m-%d %H:%M:%S.000')

                except Exception as e:
                    # If date conversion fails, leave as-is
                    logging.warning(f"Error converting dates in column {col}: {e}")

            elif sql_type.upper() in ('INT', 'BIGINT', 'SMALLINT', 'TINYINT'):
                # FIXED: Don't use pd.to_numeric() - it will treat 4.147 as 4.147 instead of 4147
                # Since Portuguese numbers are already converted, just ensure proper int conversion
                for idx in processed_df.index:
                    val = processed_df.at[idx, col]
                    if pd.isna(val) or val == '':
                        processed_df.at[idx, col] = 0
                    else:
                        try:
                            # Convert to int, handling both numeric and string values
                            if isinstance(val, (int, float)):
                                processed_df.at[idx, col] = int(val)
                            else:
                                # Should already be converted, but just in case
                                processed_df.at[idx, col] = int(float(str(val)))
                        except (ValueError, TypeError):
                            processed_df.at[idx, col] = 0

            elif sql_type.upper() in ('DECIMAL', 'NUMERIC', 'FLOAT', 'REAL'):
                # FIXED: Don't use pd.to_numeric() for the same reason
                for idx in processed_df.index:
                    val = processed_df.at[idx, col]
                    if pd.isna(val) or val == '':
                        processed_df.at[idx, col] = 0.0
                    else:
                        try:
                            # Convert to float, handling both numeric and string values
                            if isinstance(val, (int, float)):
                                processed_df.at[idx, col] = float(val)
                            else:
                                # Should already be converted, but just in case
                                processed_df.at[idx, col] = float(str(val))
                        except (ValueError, TypeError):
                            processed_df.at[idx, col] = 0.0

            elif sql_type.upper() in ('VARCHAR', 'NVARCHAR', 'CHAR', 'NCHAR', 'TEXT', 'NTEXT'):
                # Convert to strings (with NaN handling)
                processed_df[col] = processed_df[col].fillna('').astype(str)

                # Check if there's a length limit
                if '(' in sql_type and ')' in sql_type and 'MAX' not in sql_type.upper():
                    # Extract length limit
                    length_match = re.search(r'\((\d+)\)', sql_type)
                    if length_match:
                        max_length = int(length_match.group(1))
                        # Truncate strings that are too long
                        processed_df[col] = processed_df[col].str.slice(0, max_length)

        return processed_df

    def normalize_column_name(self, column_name: str, logger: Optional[logging.Logger] = None) -> str:
        """
        Enhanced column name normalization that better handles Portuguese characters
        and special cases for SQL Server compatibility.

        Args:
            column_name (str): Original column name
            logger (Optional[logging.Logger]): Logger instance

        Returns:
            str: Normalized column name suitable for SQL Server
        """
        try:
            # Portuguese character mappings
            char_mappings = {
                'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a', 'ä': 'a',
                'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
                'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
                'ó': 'o', 'ò': 'o', 'õ': 'o', 'ô': 'o', 'ö': 'o',
                'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
                'ý': 'y', 'ÿ': 'y',
                'ñ': 'n',
                'ç': 'c',
                '°': '', '²': '2', '³': '3', '€': 'eur',
                '$': 'dollar', '%': 'percent',
                '(': '', ')': '', '[': '', ']': '', '{': '', '}': '',
                '/': '_', '\\': '_', '|': '_', '-': '_', '.': '_'
            }

            # Common unit indicators to handle specially
            unit_indicators = {
                '(KB)': '',
                '(MB)': '',
                '(GB)': '',
                '(TB)': '',
                '($)': '',
                '(%)': '',
                '(#)': ''
            }

            # Convert to string and lowercase
            name = str(column_name).lower()

            # Handle unit indicators first
            for indicator, replacement in unit_indicators.items():
                if indicator.lower() in name:
                    name = name.replace(indicator.lower(), replacement)

            # Replace special characters
            for original, replacement in char_mappings.items():
                name = name.replace(original, replacement)

            # Remove any remaining diacritics
            name = ''.join(c for c in unicodedata.normalize('NFKD', name)
                        if not unicodedata.combining(c))

            # Replace any remaining non-alphanumeric chars with underscore
            name = re.sub(r'[^a-z0-9_]', '_', name)

            # Replace multiple underscores with single underscore
            name = re.sub(r'_+', '_', name)

            # Remove leading/trailing underscores
            name = name.strip('_')

            # Convert to camelCase
            parts = name.split('_')
            camel_case = parts[0] + ''.join(p.capitalize() for p in parts[1:])

            # Handle SQL Server reserved words
            sql_reserved_words = {
                'add', 'all', 'alter', 'and', 'any', 'as', 'asc', 'backup', 'begin',
                'between', 'by', 'case', 'check', 'column', 'constraint', 'create',
                'database', 'default', 'delete', 'desc', 'distinct', 'drop', 'exec',
                'exists', 'foreign', 'from', 'full', 'group', 'having', 'in', 'index',
                'inner', 'insert', 'into', 'is', 'join', 'key', 'left', 'like', 'not',
                'null', 'or', 'order', 'outer', 'primary', 'procedure', 'right', 'rownum',
                'select', 'set', 'table', 'top', 'truncate', 'union', 'unique', 'update',
                'values', 'view', 'where', 'date', 'type', 'de', 'para', 'com', 'sem',
                'ou', 'mas', 'não', 'sim', 'ainda', 'então', 'porque', 'quando', 'onde',
                'como', 'quem', 'qual', 'qualquer', 'algum', 'nenhum', 'muito', 'pouco',
                'mais', 'menos', 'tanto', 'cada', 'outro', 'mesmo', 'mesma', 'diferente',
                'mesmo', 'mesma', 'tudo', 'todos', 'todas', 'alguma', 'algumas', 'nenhum',
                'nenhuma', 'alguns', 'algumas', 'outros', 'outras', 'cada', 'cada um',
                'cada uma', 'algum', 'alguma', 'nenhum', 'nenhuma', 'muito', 'muita',
                'pouco', 'pouca', 'mais', 'menos', 'tanto', 'tanta', 'outro', 'outra',
                'mesmo', 'mesma', 'diferente', 'mesmo', 'mesma', 'tudo', 'todos', 'todas',
                'alguma', 'algumas', 'nenhum', 'nenhuma', 'alguns', 'algumas', 'outros',
                'outras', 'cada', 'cada um', 'cada uma', 'algum', 'alguma', 'nenhum',
                'nenhuma', 'muito', 'muita', 'pouco', 'pouca', 'mais', 'menos', 'tanto',
                'tanta', 'outro', 'outra', 'mesmo', 'mesma', 'diferente', 'mesmo',
                'mesma', 'tudo', 'todos', 'todas', 'alguma', 'algumas', 'nenhum',
                'nenhuma', 'alguns', 'algumas', 'outros', 'outras', 'cada', 'cada um'
            }

            if camel_case.lower() in sql_reserved_words:
                camel_case += 'Col'

            # Ensure name starts with a letter
            if not camel_case[0].isalpha():
                camel_case = 'n' + camel_case

            # Truncate to SQL Server's limit
            camel_case = camel_case[:128]

            if logger:
                logger.debug(f"Normalized column name: {column_name} -> {camel_case}")

            return camel_case

        except Exception as e:
            if logger:
                logger.error(f"Error normalizing column name '{column_name}': {str(e)}")
            # Return a safe fallback name
            return f"column_{abs(hash(str(column_name))) % 1000}"

    def create_table_if_not_exists(self, table: str, columns: List[str], values: List[Tuple[Any]]) -> Tuple[bool, Dict[str, str]]:
        """
        Create a table with normalized column names if it does not already exist.

        Args:
            table (str): The name of the table to create
            columns (List[str]): List of original column names
            values (List[Tuple[Any]]): Sample data to infer column types

        Returns:
            Tuple[bool, Dict[str, str]]:
                - bool: True if table was created, False if it already existed
                - Dict[str, str]: Mapping of original to normalized column names
        """
        try:
            # Split schema and table name
            schema_name = table.split('.')[0] if '.' in table else 'dbo'
            table_name = table.split('.')[-1]

            # Check if table exists
            check_query = """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = ?
            AND TABLE_NAME = ?
            """

            result = self.db_client.execute_query(check_query, (schema_name, table_name))
            table_exists = result[0][0] > 0 if result else False

            if table_exists:
                # Return existing column mapping
                existing_columns_query = """
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = ?
                AND TABLE_NAME = ?
                """
                existing_cols = self.db_client.execute_query(existing_columns_query, (schema_name, table_name))
                existing_mapping = {col: col for col, in existing_cols}
                return False, existing_mapping

            # Create mapping of original to normalized names
            column_mapping = {col: self.normalize_column_name(col) for col in columns}

            # Handle duplicate normalized names
            seen_names = {}
            for original, normalized in column_mapping.items():
                if normalized in seen_names:
                    count = seen_names[normalized] + 1
                    seen_names[normalized] = count
                    column_mapping[original] = f"{normalized}{count}"
                else:
                    seen_names[normalized] = 1

            # Log the column mapping
            logger.info("Column name mapping:")
            for original, normalized in column_mapping.items():
                logger.info(f"  {original} -> {normalized}")

            # Create list of values with columns in the new order
            normalized_columns = list(column_mapping.values())

            # Infer column types using normalized names
            column_types = self._infer_column_types(values, normalized_columns)

            # Create the columns definition string
            columns_def = ", ".join([f"[{col}] {dtype}" for col, dtype in column_types.items()])

            # Ensure schema exists
            create_schema_query = """
            IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = ?)
            BEGIN
                EXEC('CREATE SCHEMA [{schema_name}]')
            END
            """
            self.db_client.execute_query(create_schema_query, (schema_name,))

            # Create table query
            create_query = f"""
            CREATE TABLE [{schema_name}].[{table_name}] (
                {columns_def}
            )
            """
            self.db_client.execute_query(create_query)
            logger.info(f"Table '{table}' created successfully with normalized columns")

            return True, column_mapping

        except Exception as e:
            logger.error(f"Failed to create table '{table}'. Error: {str(e)}")
            raise

    def _get_valid_columns(self, table: str, provided_columns: List[str] = None) -> List[str]:
        """
        Get and validate non-identity columns for a table.

        Args:
            table (str): The table name (can include schema)
            provided_columns (List[str], optional): List of columns to validate against

        Returns:
            List[str]: List of valid column names in correct order
        """
        try:
            # Split schema and table name
            schema_name = table.split('.')[0] if '.' in table else 'dbo'
            table_name = table.split('.')[-1]

            # Get non-identity columns
            query = """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = ?
            AND COLUMNPROPERTY(OBJECT_ID(TABLE_NAME), COLUMN_NAME, 'IsIdentity') = 0
            ORDER BY ORDINAL_POSITION
            """

            result = self.db_client.execute_query(query, (table_name,), fetch_as_dict=True)
            actual_columns = [row['COLUMN_NAME'] for row in result]

            if not provided_columns:
                return actual_columns

            # Validate provided columns
            valid_columns = [col for col in provided_columns
                            if col.upper() in (col.upper() for col in actual_columns)]

            if len(valid_columns) != len(actual_columns):
                logger.warning(f"Column mismatch. Expected: {actual_columns}, Got: {valid_columns}")

            return valid_columns

        except Exception as e:
            logger.error(f"Error getting valid columns for {table}: {str(e)}")
            raise

    @retry(max_retries=3, delay=2, backoff=1.5, exceptions=(Exception,), logger=logger)
    def table_exists(self, table: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table (str): The name of the table to check.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        query = """
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_NAME = ?
        """

        try:
            result = self.db_client.execute_query(query, (table,), fetch_as_dict=False)
            return result[0][0] > 0
        except Exception as e:
            logger.error(f"Failed to check if table '{table}' exists. Error: {e}")
            raise

    def cleanup_values(self, values, columns, column_types=None, logger=None):
        """Enhanced data validation and cleanup with Portuguese number support"""
        from helpers.number_converter import convert_portuguese_to_english_number

        cleaned_values = []
        problem_rows = []

        for row_idx, row in enumerate(values):
            cleaned_row = []
            row_has_problems = False

            for col_idx, val in enumerate(row):
                col_name = columns[col_idx] if col_idx < len(columns) else f"column_{col_idx}"
                try:
                    # Handle None values
                    if val is None or (isinstance(val, str) and val.strip() == ''):
                        cleaned_row.append(None)
                        continue

                    # Get column type if available
                    col_type = None
                    if column_types and col_name in column_types:
                        col_type = column_types[col_name].upper()

                    # Convert based on column type
                    if col_type and 'INT' in col_type:
                        # For integer columns - use Portuguese converter
                        if isinstance(val, str):
                            val = val.strip()
                            if val == '':
                                cleaned_row.append(None)
                            else:
                                # Use Portuguese number converter instead of basic replace
                                converted = convert_portuguese_to_english_number(val)
                                if isinstance(converted, (int, float)):
                                    cleaned_row.append(int(converted))
                                else:
                                    cleaned_row.append(0)  # Fallback
                        else:
                            cleaned_row.append(int(val) if val is not None else None)

                    elif col_type and any(t in col_type for t in ['FLOAT', 'DECIMAL', 'NUMERIC', 'REAL']):
                        # For float columns - use Portuguese converter
                        if isinstance(val, str):
                            val = val.strip()
                            if val == '':
                                cleaned_row.append(None)
                            else:
                                # Use Portuguese number converter instead of basic replace
                                converted = convert_portuguese_to_english_number(val)
                                if isinstance(converted, (int, float)):
                                    cleaned_row.append(float(converted))
                                else:
                                    cleaned_row.append(0.0)  # Fallback
                        else:
                            cleaned_row.append(float(val) if val is not None else None)

                    elif col_type and any(t in col_type for t in ['DATE', 'DATETIME']):
                        # For date columns
                        if isinstance(val, (datetime, date)):
                            cleaned_row.append(val)
                        elif isinstance(val, str):
                            val = val.strip()
                            if val == '':
                                cleaned_row.append(None)
                            else:
                                # Try multiple date formats
                                for fmt in ['%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y']:
                                    try:
                                        date_val = datetime.strptime(val, fmt)
                                        cleaned_row.append(date_val)
                                        break
                                    except ValueError:
                                        continue
                                else:
                                    # If no format works, log it and use None
                                    if logger:
                                        logger.warning(f"Invalid date value '{val}' in column '{col_name}' at row {row_idx+1}")
                                    cleaned_row.append(None)
                                    row_has_problems = True
                        else:
                            cleaned_row.append(None)

                    else:
                        # For string/other columns - ensure they're properly encoded strings
                        if isinstance(val, str):
                            # Replace null bytes and other problematic characters
                            cleaned_str = val.replace('\x00', '').replace('\x1a', '')
                            cleaned_row.append(cleaned_str)
                        else:
                            cleaned_row.append(str(val) if val is not None else None)

                except Exception as e:
                    if logger:
                        logger.warning(f"Error processing value '{val}' in column '{col_name}' at row {row_idx+1}: {str(e)}")
                    cleaned_row.append(None)
                    row_has_problems = True

            cleaned_values.append(tuple(cleaned_row))
            if row_has_problems:
                problem_rows.append(row_idx)

        if logger and problem_rows:
            logger.warning(f"Found {len(problem_rows)} problematic rows: {problem_rows[:10]}...")

        return cleaned_values

    def deep_clean_values(self, values: List[Tuple[Any]], columns: List[str]) -> List[Tuple[Any]]:
        """
        Thoroughly clean all values to ensure they are compatible with SQL Server.

        Args:
            values (List[Tuple[Any]]): List of value tuples
            columns (List[str]): Column names matching the values

        Returns:
            List[Tuple[Any]]: Cleaned values
        """
        import re
        from datetime import datetime, date

        cleaned_values = []

        for row_idx, row in enumerate(values):
            cleaned_row = []

            for col_idx, val in enumerate(row):
                col_name = columns[col_idx] if col_idx < len(columns) else f"column_{col_idx}"

                try:
                    # Handle None values
                    if val is None:
                        cleaned_row.append(None)
                        continue

                    # Convert empty strings to None
                    if isinstance(val, str) and val.strip() == '':
                        cleaned_row.append(None)
                        continue

                    # First, check column name hints for data types
                    col_lower = col_name.lower()

                    # Handle special characters in string values
                    if isinstance(val, str):
                        # Remove null bytes and control characters
                        val = val.replace('\x00', '').replace('\x1a', '')

                        # Handle different types based on column name hints
                        if any(hint in col_lower for hint in ['id', 'code', 'num', 'quantidade', 'qtd', 'stock']):
                            # Numeric columns - strip any non-numeric chars except decimal separator
                            numeric_str = re.sub(r'[^0-9.,\-+]', '', val.replace(',', '.'))
                            if numeric_str.strip():
                                try:
                                    # Try to convert to numeric
                                    numeric_val = float(numeric_str)
                                    # Check if it should be integer
                                    if numeric_val.is_integer():
                                        cleaned_row.append(int(numeric_val))
                                    else:
                                        cleaned_row.append(numeric_val)
                                except ValueError:
                                    # If conversion fails, use None
                                    cleaned_row.append(None)
                            else:
                                cleaned_row.append(None)

                        elif any(hint in col_lower for hint in ['data', 'date', 'dt', 'dia', 'entrada', 'lancamento']):
                            # Date columns - try various formats
                            if val.strip():
                                for fmt in ['%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y']:
                                    try:
                                        date_val = datetime.strptime(val, fmt)
                                        cleaned_row.append(date_val.strftime('%Y-%m-%d'))
                                        break
                                    except ValueError:
                                        continue
                                else:
                                    # If no format works, use None
                                    cleaned_row.append(None)
                            else:
                                cleaned_row.append(None)

                        else:
                            # Regular string columns - ensure UTF-8 encoding
                            try:
                                # Try to clean the string by decoding/encoding
                                clean_str = val.encode('utf-8', errors='ignore').decode('utf-8')
                                cleaned_row.append(clean_str[:8000])  # Limit length for SQL Server
                            except Exception:
                                # If encoding fails, use a stripped version
                                cleaned_row.append(str(val).strip()[:8000])

                    # Handle dates and datetimes
                    elif isinstance(val, (date, datetime)):
                        if isinstance(val, datetime):
                            cleaned_row.append(val.strftime('%Y-%m-%d %H:%M:%S'))
                        else:
                            cleaned_row.append(val.strftime('%Y-%m-%d'))

                    # Handle numeric values
                    elif isinstance(val, (int, float)):
                        cleaned_row.append(val)

                    # Handle any other types
                    else:
                        cleaned_row.append(str(val))

                except Exception as e:
                    logger.warning(f"Error cleaning value in row {row_idx}, column '{col_name}': {str(e)}")
                    cleaned_row.append(None)

            cleaned_values.append(tuple(cleaned_row))

        return cleaned_values

    def create_without_progress(self, table: str, values: List[Tuple[Any]], columns: List[str] = None) -> bool:
        """
        Create new records in the specified table without showing a progress bar.
        Useful for batch operations and debugging.

        Args:
            table (str): The table name
            values (List[Tuple[Any]]): List of tuples of values to insert
            columns (List[str], optional): List of column names. If None, columns will be inferred

        Returns:
            bool: True if records were inserted successfully
        """
        if columns is None:
            columns = self._get_table_columns(table)

        if not isinstance(values, list):
            values = [values]
        elif not all(isinstance(v, tuple) for v in values):
            raise ValueError("Values must be a tuple or a list of tuples.")

        # Validate data format
        for value_tuple in values:
            if len(value_tuple) != len(columns):
                raise ValueError(f"Number of values {len(value_tuple)} does not match number of columns {len(columns)}")

        # Create the table if it doesn't exist and get column mapping
        _, column_mapping = self.create_table_if_not_exists(table, columns, values)

        # Get the normalized column names
        normalized_columns = [col for col in column_mapping.values()]

        # Get valid columns
        valid_columns = self._get_valid_columns(table, normalized_columns)

        # Create the INSERT query
        columns_str = ", ".join([f"[{col}]" for col in valid_columns])
        placeholders = ", ".join(["?"] * len(valid_columns))
        query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"

        # Clean the values
        cleaned_values = self.deep_clean_values(values, columns)

        try:
            # Execute batch query without progress bar
            self.db_client.execute_batch_query(query, cleaned_values)
            return True
        except Exception as e:
            logger.error(f"Failed to insert records: {str(e)}")
            return False

    @retry(max_retries=5, delay=5, backoff=2, exceptions=(Exception,), logger=logger)
    def create(self, table: str, values: List[Tuple[Any]], columns: List[str] = None, batch_size: int = 1000, show_progress: bool = True) -> bool:
        """
        Create new records in the specified table with enhanced type handling and error detection.

        Args:
            table (str): The table name
            values (List[Tuple[Any]]): List of tuples of values to insert
            columns (List[str], optional): List of column names. If None, columns will be inferred

        Returns:
            bool: True if records were inserted successfully
        """
        if columns is None:
            columns = self._get_table_columns(table)

        if not isinstance(values, list):
            values = [values]
        elif not all(isinstance(v, tuple) for v in values):
            raise ValueError("Values must be a tuple or a list of tuples.")

        # Validate data
        for value_tuple in values:
            if len(value_tuple) != len(columns):
                raise ValueError(f"Number of values {len(value_tuple)} does not match number of columns {len(columns)}")

        # Create the table if it doesn't exist and get column mapping
        _, column_mapping = self.create_table_if_not_exists(table, columns, values)

        # Get the normalized column names from keys
        normalized_columns = [col for col in column_mapping.values()]

        # Get valid columns using helper method
        valid_columns = self._get_valid_columns(table, normalized_columns)

        # Create the INSERT query with normalized column names
        columns_str = ", ".join([f"[{col}]" for col in valid_columns])
        placeholders = ", ".join(["?" for _ in valid_columns])
        query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"

        # cleanup values
        string_values = self.cleanup_values(values, columns, column_types=None, logger=logger)

        # Disable logging temporarily if showing progress bar
        if show_progress:
            logger.disabled = True

        try:
            if show_progress:
                # Process in batches with a single progress bar
                with tqdm(total=len(string_values), desc=f"Inserting into {table}") as pbar:
                    for i in range(0, len(string_values), batch_size):
                        batch = string_values[i:i+batch_size]
                        self.db_client.execute_batch_query(query, batch)
                        pbar.update(len(batch))
            else:
                # Process without progress bar for automated processes
                for i in range(0, len(string_values), batch_size):
                    batch = string_values[i:i+batch_size]
                    self.db_client.execute_batch_query(query, batch)

            return True
        except Exception as e:
            logger.disabled = False  # Re-enable logging
            logger.error(f"Failed to insert records: {e}")
            return False

        finally:
            logger.disabled = False  # Make sure logging is re-enabled

    @retry(max_retries=5, delay=5, backoff=2, exceptions=(Exception,), logger=logger)
    def read(self, table: str, columns: List[str] = None, where: str = "", params: Tuple[Any] = None, show_id: bool = False, batch_size: int = None) -> List[Dict[str, Any]]:
        """
        Read records from the specified table with optional batch support.

        Args:
            table (str): The table name.
            columns (list, optional): List of column names to retrieve. If None, all columns will be retrieved.
            where (str, optional): WHERE clause for filtering records.
            params (tuple, optional): Tuple of parameters for the WHERE clause.
            show_id (bool, optional): If True, include the 'id' column. Default is False.
            batch_size (int, optional): If provided, limits the number of records returned per batch.

        Returns:
            list: List of records as dictionaries.
        """
        if columns is None:
            columns = self._get_table_columns(table, show_id=show_id)

        columns_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {columns_str} FROM {table}"
        if where:
            query += f" WHERE {where}"
        if batch_size:
            query += f" ORDER BY {columns[0]} OFFSET 0 ROWS FETCH NEXT {batch_size} ROWS ONLY"

        try:
            result = self.db_client.execute_query(query, params, fetch_as_dict=True)
            records = [self._format_dates(row) for row in result]
            logger.info(f"Records read successfully, {len(records)} rows found.")
            return records
        except Exception as e:
            logger.error(f"Failed to read records. Error: {e}")
            raise

    @retry(max_retries=5, delay=5, backoff=2, exceptions=(Exception,), logger=logger)
    def update(self, table: str, updates: Dict[str, Any], where: str, params: Tuple[Any]) -> bool:
        """
        Update records in the specified table.

        Args:
            table (str): The table name.
            updates (dict): Dictionary of columns and their new values.
            where (str): WHERE clause for identifying records to update.
            params (tuple): Tuple of parameters for the WHERE clause.

        Returns:
            bool: True if records were updated successfully, False otherwise.
        """
        set_clause = ", ".join([f"{col} = ?" for col in updates.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {where}"
        values = tuple(updates.values()) + params
        try:
            self.db_client.execute_query(query, values)
            logger.info("Records updated successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to update records. Error: {e}")
            return False

    @retry(max_retries=5, delay=5, backoff=2, exceptions=(Exception,), logger=logger)
    def delete(self, table: str, where: str = "", params: Tuple[Any] = None, batch_size: int = None) -> bool:
        """
        Delete records from the specified table with optional batch processing.

        Args:
            table (str): The table name.
            where (str, optional): WHERE clause for identifying records to delete. If empty, all records will be deleted.
            params (tuple, optional): Tuple of parameters for the WHERE clause.
            batch_size (int, optional): If provided, deletes records in batches.

        Returns:
            bool: True if records were deleted successfully, False otherwise.
        """

        # Check if the table exists before attempting to delete
        if not self.table_exists(table):
            logger.warning(f"Table '{table}' does not exist. Delete operation aborted.")
            return False

        query = f"DELETE FROM {table}"
        if where:
            query += f" WHERE {where}"
        if batch_size:
            query += f" ORDER BY (SELECT NULL) OFFSET 0 ROWS FETCH NEXT {batch_size} ROWS ONLY"

        try:
            self.db_client.execute_query(query, params)
            logger.info("Records deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete records. Error: {e}")
            return False

    def execute_raw_query(self, query: str, params: Optional[Dict[str, Any]] = None, fetch_as_dict: bool = True) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a raw SQL query with improved safety and result handling.

        Args:
            query (str): The SQL query.
            params (dict, optional): Query parameters.
            fetch_as_dict (bool): Return results as dictionaries.

        Returns:
            Optional[list]: Query results or None for non-SELECT queries.
        """
        try:
            is_select = query.strip().lower().startswith('select')
            result = self.db_client.execute_query(query, params, fetch_as_dict=fetch_as_dict)

            if is_select and fetch_as_dict:
                return [self._format_dates(record) for record in result]
            return result
        except Exception as e:
            logger.error(f"Failed to execute raw query: {e}")
            raise
