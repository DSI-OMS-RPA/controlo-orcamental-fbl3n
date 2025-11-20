import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import re
import unicodedata
from dotenv import load_dotenv

import pandas as pd
from tqdm import tqdm

from helpers.configuration import load_ini_config, load_json_config
from helpers.database.database_factory import DatabaseFactory
from helpers.database.postgresql_generic_crud import PostgresqlGenericCRUD
from helpers.database.sqlserver_generic_crud import SQLServerGenericCRUD
from helpers.utils import setup_logger

# Load environment variables from a .env file
load_dotenv()

# Setup logger
logger = setup_logger(__name__)

# Configuration Data Class
@dataclass
class ETLConfig:
    table: str
    table_control: str
    error_report: dict
    smtp_configs: dict
    client_accounts: dict
    report: dict
    sap_app: dict
    database: dict
    process: dict

def create_alert_message(title, start_date, all_success, any_partial):
    """
    Create an alert message based on the processing results.

    Args:
        title (str): The title of the alert.
        start_date (datetime): The start date of the processing period.
        all_success (bool): True if all processing tasks were successful.
        any_partial (bool): True if some processing tasks were successful.

    Returns:
        dict: A dictionary containing the alert type, title, and message
    """

    if all_success:
        return {
            'alert_type': 'success',
            'alert_title': title,
            'alert_message': f'O processo foi executado com sucesso em {start_date:%Y-%m-%d}. Favor verificar.'
        }
    elif any_partial:
        return {
            'alert_type': 'warning',
            'alert_title': title,
            'alert_message': f'O processo foi parcialmente executado em {start_date:%Y-%m-%d}. Verifique o log de execução.'
        }
    else:
        return {
            'alert_type': 'danger',
            'alert_title': title,
            'alert_message': 'Ocorreu um erro ao salvar os dados, favor verificar o log de execução.'
        }

def load_etl_config() -> ETLConfig:
    """
    Load the ETL configuration settings from the configuration files.
    Returns:
        ETLConfig: An instance of the ETLConfig data class with the loaded configuration settings.
    """

    db_config = load_ini_config("DATABASE")
    report = load_json_config().get("report")
    process = load_json_config().get("process")
    database = load_json_config().get("database")
    sap_app = load_json_config().get("sap_app")
    client_accounts = load_json_config().get("client_number")
    error_report = load_json_config().get("error_report")
    smtp_configs = load_ini_config("SMTP")

    return ETLConfig(
        table=db_config.get("table"),
        table_control=db_config.get("table_control"),
        error_report=error_report,
        smtp_configs=smtp_configs,
        report=report,
        sap_app=sap_app,
        database=database,
        process=process,
        client_accounts=client_accounts
    )

async def setup_db_connections():
    """
    Set up and establish connections to various databases used in the ETL process.

    Returns:
        Tuple containing:
        - SQL Server connection and CRUD object
        - PostgreSQL connection and CRUD object
    """
    retries = 3
    delay = 5

    for attempt in range(retries):
        try:
            logger.info("Setting up database connections...")
            config = {
                'dmkbi': load_ini_config("CVTVMDWBI"),
                'postgresql': load_ini_config("POSTGRESQL")
            }

            # Create the database connection objects
            dmkbi_db = DatabaseFactory.get_database('sqlserver', config['dmkbi'])
            postgresql_db = DatabaseFactory.get_database('postgresql', config['postgresql'])

            # Open all connections manually
            logger.info("Connecting to SQL Server and PostgreSQL databases...")
            dmkbi_db.connect()
            postgresql_db.connect()

            # Create CRUD objects for respective databases
            dmkbi_crud = SQLServerGenericCRUD(dmkbi_db)
            postgresql_crud = PostgresqlGenericCRUD(postgresql_db)

            logger.info("Database connections established successfully.")
            return dmkbi_db, dmkbi_crud, postgresql_db, postgresql_crud

        except Exception as e:
            logger.error(f"Error setting up database connections (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                raise


def close_connections(dmkbi_db, postgresql_db):
    """
    Close connections to the SQL Server and PostgreSQL databases.

    Safe to call multiple times - checks connection state before closing.
    """
    try:
        logger.info("Closing database connections...")

        if dmkbi_db:
            try:
                dmkbi_db.disconnect()
                logger.info("SQL Server connection closed.")
            except Exception as e:
                logger.warning(f"Error closing SQL Server connection: {e}")

        if postgresql_db:
            try:
                postgresql_db.disconnect()
                logger.info("PostgreSQL connection closed.")
            except Exception as e:
                logger.warning(f"Error closing PostgreSQL connection: {e}")

    except Exception as e:
        logger.error(f"Unexpected error in close_connections: {e}")


@asynccontextmanager
async def managed_resources():
    """
    A context manager to manage resources for the ETL process.

    Guarantees that resources are cleaned up exactly once, even if exceptions occur.
    The caller should NOT manually close connections - the context manager handles it.

    Yields:
        Tuple: (config, dmkbi_crud, postgresql_crud, (dmkbi_db, postgresql_db))

    Usage:
        async with managed_resources() as (config, dmkbi_crud, postgresql_crud, dbs):
            # Use resources here
            # Connections are automatically closed when exiting this block
            pass
    """

    dmkbi_db = None
    postgresql_db = None
    dmkbi_crud = None
    postgresql_crud = None
    config = None

    try:
        # Step 1: Load configuration
        config = load_etl_config()
        logger.info("ETL configuration loaded successfully.")

        # Step 2: Setup database connections
        dmkbi_db, dmkbi_crud, postgresql_db, postgresql_crud = await setup_db_connections()

        # Step 3: Yield resources to caller
        yield config, dmkbi_crud, postgresql_crud, (dmkbi_db, postgresql_db)

    except Exception as e:
        logger.error(f"Error in managed_resources context manager: {e}")
        raise

    finally:
        # Cleanup happens exactly once here
        logger.info("Cleaning up resources...")
        close_connections(dmkbi_db, postgresql_db)

def get_last_processed_date(postgresql_crud, table: str, process_name: str) -> datetime:
    """
    Get the last processed date from the control table.

    Args:
        postgresql_crud: PostgreSQL CRUD handler

    Returns:
        datetime: The last processed date, or None if no record exists
    """
    try:
        logger.info("Getting last processed date from control table")

        yesterday = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)

        # Query the control table to get the last processed date
        result = postgresql_crud.read(
            table,
            ['last_processed_date'],
            where="process_name = %s",
            params=(process_name,)
        )

        if result and result[0].get('last_processed_date'):
            last_date = datetime.strptime(result[0]['last_processed_date'], '%Y-%m-%d')
            logger.info(f"Last processed date retrieved: {last_date}")
            # Return the day after the last processed date
            return last_date + timedelta(days=1)
        else:
            logger.warning("No last processed date found, using default start date")
            # Return a default date if no record exists
            return yesterday  # Default start date

    except Exception as e:
        logger.error(f"Error retrieving last processed date: {e}")
        # Return a default date if an error occurs
        return yesterday  # Default start date

def update_last_processed_date(postgresql_crud, end_date, table: str, process_name: str) -> bool:
    """
    Update the last processed date in the control table.
    Args:
        postgresql_crud: PostgreSQL CRUD handler
        end_date (datetime): The date to set as last processed
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        logger.info(f"Updating last processed date to {end_date}")
        # Check if a record already exists
        result = postgresql_crud.read(
            table,
            ['id'],
            where="process_name = %s",
            params=(process_name,)
        )

        # Format the date as string in the expected format
        date_str = end_date.strftime('%Y-%m-%d')

        if result:
            # Update existing record
            success = postgresql_crud.update(
                table,
                {'last_processed_date': date_str, 'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                where="process_name = %s",
                params=(process_name,)
            )
        else:
            # Get next ID value from sequence
            id_result = postgresql_crud.execute_raw_query(f"SELECT nextval('{table}_id_seq')")
            next_id = id_result[0]['nextval']

            # Create new record with explicit ID
            success = postgresql_crud.create(
                table,
                [(
                    next_id,
                    process_name,
                    date_str,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )],
                ['id', 'process_name', 'last_processed_date', 'created_at', 'updated_at']
            )

        return success
    except Exception as e:
        logger.error(f"Error updating last processed date: {e}")
        return False

def ensure_control_table_exists(postgresql_crud, table: str) -> None:
    """
    Ensure the ETL control table exists in the database.

    Args:
        postgresql_crud: PostgreSQL CRUD handler
        table: Name of the control table to create
    """
    try:
        # Check if table exists by trying to read from it
        postgresql_crud.read(table, ['id'], where="1=0")
        logger.info(f"ETL control table {table} already exists")
    except Exception:
        # Create the table if it doesn't exist
        logger.info(f"Creating ETL control table {table}")

        # Use an f-string to format the table name
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id SERIAL PRIMARY KEY,
            process_name VARCHAR(50) NOT NULL UNIQUE,
            last_processed_date DATE NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
        postgresql_crud.execute_raw_query(create_table_query)
        logger.info(f"ETL control table {table} created successfully")

def save_with_progress(crud, table, values, columns=None, batch_size=1000):
    """
    Save data to database with progress bar.

    Args:
        crud: Database CRUD handler
        table (str): Table name
        values (List[Tuple]): Data to save
        columns (List[str], optional): Column names
        batch_size (int): Size of each batch

    Returns:
        bool: True if successful
    """
    total_rows = len(values)

    with tqdm(total=total_rows, desc=f"Saving to {table}") as pbar:
        for i in range(0, total_rows, batch_size):
            batch = values[i:i+batch_size]
            # Create database record without progress bar
            result = crud.create_without_progress(table, batch)
            if not result:
                return False
            pbar.update(len(batch))

    return True

def handle_no_data_found(
    postgresql_crud,
    control_table,
    process_name,
    raw_name,
    start_date,
    email_sender=None,
    report_config=None,
    send_email=False
):
    """
    Handle scenario when no data is found in SAP report.

    This function:
    1. Updates the last processed date in the control table
    2. Creates an alert message
    3. Sends a notification email if enabled
    4. Logs the completion status

    Args:
        postgresql_crud: Database CRUD handler for PostgreSQL
        end_date (datetime): End date of the processing period
        control_table (str): Name of the control table
        process_name (str): Name of the ETL process
        raw_name (str): Raw name of the process (for alert message)
        start_date (datetime): Start date of the processing period
        email_sender (EmailSender, optional): Email sender instance
        report_config (dict, optional): Email report configuration
        send_email (bool): Whether to send email notifications

    Returns:
        bool: True if the handling was successful
    """
    logger.warning("File contains 'Lista não contém dados' message. No data to process.")

    # Update the last processed date even though no data was processed
    success = update_last_processed_date(postgresql_crud, start_date, control_table, process_name)

    # Send notification about no data found
    alert_info = create_alert_message(
        raw_name,
        start_date,
        True,  # Success status true since this is a valid "no data" scenario
        False
    )

    if send_email and email_sender and report_config:
        # Send email report about no data
        email_sent = email_sender.send_template_email(
            report_config=report_config,
            alert_type='warning',  # Use warning for no data scenarios
            alert_title=f"{alert_info['alert_title']} - Sem dados",
            alert_message=f"Nenhum dado encontrado no relatório SAP em {start_date.strftime('%Y-%m-%d')}.",
            environment='production',
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        if email_sent:
            logger.info("No data notification email sent successfully.")
        else:
            logger.warning("Failed to send no data notification email.")

    logger.info("ETL process completed successfully with no data to process.")
    return success

def debug_data_batch(df, batch_start=10900, batch_end=11100, verbose=True):
    """
    Debug a specific batch of data from the DataFrame to identify problematic values.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data
        batch_start (int): Starting index of the batch to debug
        batch_end (int): Ending index of the batch to debug
        verbose (bool): Whether to print detailed debugging information

    Returns:
        pandas.DataFrame: DataFrame containing only the problematic rows
    """
    import pandas as pd
    import re
    from datetime import datetime

    logger.info(f"Debugging batch from index {batch_start} to {batch_end}")

    # Extract the batch for debugging
    batch_df = df.iloc[batch_start:batch_end].copy()

    # Print basic info about the batch
    logger.info(f"Batch shape: {batch_df.shape}")
    logger.info(f"Column types: {batch_df.dtypes}")

    # Function to check if a value could cause SQL Server issues
    def is_problematic(val):
        if pd.isna(val):
            return False

        if isinstance(val, str):
            # Check for null bytes, control characters, and other problematic chars
            if '\x00' in val or '\x1a' in val:
                return True

            # Check for strings that can't be cast to numbers when in numeric columns
            if re.search(r'[^0-9.,\-+]', val) and any(c.isdigit() for c in val):
                return True

            # Check for invalid date formats in date-like strings
            if re.search(r'\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}', val):
                try:
                    # Try common date formats
                    for fmt in ['%d.%m.%Y', '%Y-%m-%d', '%d/%m/%Y']:
                        datetime.strptime(val, fmt)
                        return False
                except ValueError:
                    return True

        return False

    # Track problematic rows
    problem_rows = []

    # Check each row for problematic values
    for i, row in batch_df.iterrows():
        row_problems = {}
        for col, val in row.items():
            if is_problematic(val):
                row_problems[col] = str(val)

        if row_problems:
            problem_rows.append({
                'row_index': i,
                'problems': row_problems
            })

    # Print the problematic rows
    if problem_rows:
        logger.warning(f"Found {len(problem_rows)} problematic rows in batch")
        for idx, row_info in enumerate(problem_rows):
            if verbose:
                logger.warning(f"Problem {idx+1}: Row {row_info['row_index']}")
                for col, val in row_info['problems'].items():
                    logger.warning(f"  Column '{col}': {val}")
    else:
        logger.info("No obvious problematic values found in batch")

    # Create a DataFrame of just the problematic rows
    problem_indices = [row['row_index'] for row in problem_rows]
    return df.loc[problem_indices] if problem_indices else pd.DataFrame()

# Add this code to your main.py file, after loading the CSV and before the database insertion
def debug_and_fix_data(df):
    """
    Debug and fix problematic data in the DataFrame

    Args:
        df (pandas.DataFrame): The DataFrame to debug and fix

    Returns:
        pandas.DataFrame: The fixed DataFrame
    """
    # First identify the problematic rows in the batch where the error occurs
    problem_df = debug_data_batch(df)

    # Create a cleaned copy of the DataFrame
    cleaned_df = df.copy()

    # Fix known issues based on the debugging results

    # 1. Replace null bytes and control characters in all string columns
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        cleaned_df[col] = cleaned_df[col].astype(str).apply(
            lambda x: x.replace('\x00', '').replace('\x1a', '')
            if isinstance(x, str) else x
        )

    # 2. Handle numeric columns with potential non-numeric characters
    for col in cleaned_df.select_dtypes(include=['float64', 'int64']).columns:
        # Convert to string, clean, then convert back to numeric
        cleaned_df[col] = cleaned_df[col].astype(str).apply(
            lambda x: re.sub(r'[^0-9.,\-+]', '', x.replace(',', '.'))
            if isinstance(x, str) else x
        )
        # Convert to numeric with errors='coerce' to handle any remaining issues
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').fillna(0)

    # Log the results of the cleaning operation
    logger.info(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")

    return cleaned_df

# Modified batch processing for database insertion
def insert_with_smaller_batches(crud, table, df, columns=None, batch_size=100):
    """
    Insert data into the database with smaller batch sizes and better error handling

    Args:
        crud: Database CRUD handler
        table (str): Table name
        df (pandas.DataFrame): DataFrame to insert
        columns (list, optional): Column names
        batch_size (int): Size of each batch

    Returns:
        bool: True if successful
    """
    if columns is None:
        columns = list(df.columns)

    # Convert DataFrame to list of tuples
    values = [tuple(row) for row in df.values]
    total_rows = len(values)

    success_count = 0
    failed_batches = []

    # Process in smaller batches
    with tqdm(total=total_rows, desc=f"Inserting into {table}") as pbar:
        for i in range(0, total_rows, batch_size):
            batch = values[i:i+min(batch_size, total_rows-i)]
            batch_range = f"{i}-{i+len(batch)-1}"

            try:
                # Try to insert this batch
                result = crud.create_without_progress(table, batch, columns)
                if result:
                    success_count += len(batch)
                    pbar.update(len(batch))
                else:
                    logger.warning(f"Batch {batch_range} returned False but didn't raise exception")
                    failed_batches.append((i, i+len(batch)-1))
            except Exception as e:
                logger.error(f"Error inserting batch {batch_range}: {str(e)}")

                # Log more details about the problematic batch
                if len(batch) <= 5:  # Only log if batch is small enough
                    for idx, row in enumerate(batch):
                        logger.error(f"Problem row {i+idx}: {row}")

                failed_batches.append((i, i+len(batch)-1))

    # Log summary
    logger.info(f"Insertion complete. {success_count}/{total_rows} rows successfully inserted.")
    if failed_batches:
        logger.error(f"Failed batches: {failed_batches}")
        return False
    return True

def debug_problematic_batch(df, start_index=11700, end_index=11799):
    """
    Function to specifically debug the problematic batch identified in the logs.

    This function will analyze each row and column in the problematic batch,
    trying to identify exactly which values are causing the data type conversion error.

    Args:
        df (pandas.DataFrame): The DataFrame with all data
        start_index (int): Starting index of the problematic batch
        end_index (int): Ending index of the problematic batch

    Returns:
        pandas.DataFrame: A subset with only the data that caused issues
    """
    import pandas as pd
    import re
    from datetime import datetime

    # Extract the problematic batch
    batch_df = df.iloc[start_index:end_index+1].copy()

    # Print information about the batch
    print(f"Debugging batch from index {start_index} to {end_index}")
    print(f"Batch shape: {batch_df.shape}")

    # For each column, check all rows for potential issues
    problematic_rows = []

    for col in batch_df.columns:
        print(f"\nAnalyzing column: {col}")
        print(f"  Data type: {batch_df[col].dtype}")

        # Check for non-standard characters that might cause issues
        if batch_df[col].dtype == 'object':  # String columns
            # Look for null bytes or other problematic characters
            for idx, val in batch_df[col].items():
                if isinstance(val, str):
                    # Check for various problematic characters
                    if '\x00' in val or '\x1a' in val:
                        print(f"  Row {idx}: Found null byte or control character: {repr(val)}")
                        problematic_rows.append(idx)

                    # Check for non-printable characters
                    non_printable = [char for char in val if ord(char) < 32 or ord(char) > 126]
                    if non_printable:
                        print(f"  Row {idx}: Found non-printable characters: {repr(val)}")
                        problematic_rows.append(idx)

                    # Check for extremely long strings
                    if len(val) > 4000:
                        print(f"  Row {idx}: Very long string ({len(val)} chars)")
                        problematic_rows.append(idx)

        # Check for mismatched types - strings in numeric columns
        elif batch_df[col].dtype in ('int64', 'float64'):  # Numeric columns
            for idx, val in batch_df[col].items():
                if isinstance(val, str) and not all(c.isdigit() or c in '.-+' for c in val):
                    print(f"  Row {idx}: Non-numeric value in numeric column: {repr(val)}")
                    problematic_rows.append(idx)

        # Check for invalid date formats
        if any(date_hint in col.lower() for date_hint in ['data', 'date', 'dt']):
            for idx, val in batch_df[col].items():
                if isinstance(val, str) and re.search(r'\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}', val):
                    try:
                        # Try common date formats
                        for fmt in ['%d.%m.%Y', '%Y-%m-%d', '%d/%m/%Y']:
                            try:
                                datetime.strptime(val, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            print(f"  Row {idx}: Invalid date format: {repr(val)}")
                            problematic_rows.append(idx)
                    except Exception as e:
                        print(f"  Row {idx}: Error processing date: {repr(val)}, Error: {str(e)}")
                        problematic_rows.append(idx)

    # Get unique problematic rows
    unique_problematic_rows = list(set(problematic_rows))
    print(f"\nFound {len(unique_problematic_rows)} potentially problematic rows.")

    if unique_problematic_rows:
        print("Problematic row indices:", sorted(unique_problematic_rows))

        # Return a dataframe with only the problematic rows
        return df.loc[unique_problematic_rows]
    else:
        # If no specific issues found, output a sample from the batch for manual inspection
        print("\nNo obvious issues found. Showing first few rows for manual inspection:")
        print(batch_df.head(5))
        return pd.DataFrame()  # Return empty dataframe

def insert_with_enhanced_error_handling(crud, table, df, columns=None, batch_size=100,
                                      problematic_batch_index=11700, problematic_batch_size=100):
    """
    Insert data into database with special handling for the known problematic batch.

    This function will process batches normally, but for the known problematic batch,
    it will attempt to insert records one by one to identify and skip problematic records.

    Args:
        crud: Database CRUD handler
        table (str): Table name
        df (pandas.DataFrame): DataFrame to insert
        columns (list, optional): Column names
        batch_size (int): Size of each normal batch
        problematic_batch_index (int): Index where the known problematic batch starts
        problematic_batch_size (int): Size of the problematic batch

    Returns:
        bool: True if successful overall (even with skipped rows)
    """
    from tqdm import tqdm

    if columns is None:
        columns = list(df.columns)

    # Convert DataFrame to list of tuples
    values = [tuple(row) for row in df.values]
    total_rows = len(values)

    success_count = 0
    skipped_rows = []

    # Process in batches with special handling for the known problematic batch
    with tqdm(total=total_rows, desc=f"Inserting into {table}") as pbar:
        batch_start = 0

        while batch_start < total_rows:
            # Check if this is the problematic batch
            if batch_start == problematic_batch_index:
                print(f"\nProcessing known problematic batch ({batch_start}-{batch_start+problematic_batch_size-1}) row by row")

                # Process the problematic batch row by row
                for i in range(batch_start, min(batch_start + problematic_batch_size, total_rows)):
                    row_tuple = values[i]
                    try:
                        # Try to insert one row at a time
                        result = crud.create_without_progress(table, [row_tuple], columns)
                        if result:
                            success_count += 1
                            pbar.update(1)
                        else:
                            skipped_rows.append(i)
                            pbar.update(1)  # Still update progress even for skipped rows
                    except Exception as e:
                        print(f"Error inserting row {i}: {str(e)}")
                        skipped_rows.append(i)
                        pbar.update(1)  # Still update progress

                # Move to the next batch
                batch_start += problematic_batch_size
            else:
                # Process normal batches
                current_batch_size = min(batch_size, total_rows - batch_start)
                batch = values[batch_start:batch_start + current_batch_size]

                try:
                    # Try to insert this batch
                    result = crud.create_without_progress(table, batch, columns)
                    if result:
                        success_count += current_batch_size
                        pbar.update(current_batch_size)
                    else:
                        # If batch insertion fails without exception, try row by row
                        print(f"\nBatch {batch_start}-{batch_start+current_batch_size-1} failed, trying row by row")
                        for i, row_tuple in enumerate(batch):
                            row_index = batch_start + i
                            try:
                                result = crud.create_without_progress(table, [row_tuple], columns)
                                if result:
                                    success_count += 1
                                    pbar.update(1)
                                else:
                                    skipped_rows.append(row_index)
                                    pbar.update(1)
                            except Exception as e:
                                print(f"Error inserting row {row_index}: {str(e)}")
                                skipped_rows.append(row_index)
                                pbar.update(1)
                except Exception as e:
                    print(f"Error inserting batch {batch_start}-{batch_start+current_batch_size-1}: {str(e)}")
                    # Try row by row for this batch
                    for i, row_tuple in enumerate(batch):
                        row_index = batch_start + i
                        try:
                            result = crud.create_without_progress(table, [row_tuple], columns)
                            if result:
                                success_count += 1
                                pbar.update(1)
                            else:
                                skipped_rows.append(row_index)
                                pbar.update(1)
                        except Exception as e:
                            print(f"Error inserting row {row_index}: {str(e)}")
                            skipped_rows.append(row_index)
                            pbar.update(1)

                # Move to the next batch
                batch_start += current_batch_size

    # Log summary
    print(f"Insertion complete. {success_count}/{total_rows} rows successfully inserted.")
    if skipped_rows:
        print(f"Skipped {len(skipped_rows)} problematic rows.")
        print(f"First few skipped rows: {skipped_rows[:10]}")

        # Save skipped row data to CSV for analysis
        if len(skipped_rows) > 0:
            skipped_df = df.iloc[skipped_rows]
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"skipped_records_{timestamp}.csv"
            skipped_df.to_csv(filename, index=True)
            print(f"Saved {len(skipped_rows)} skipped records to {filename}")

    return success_count == total_rows

def ultra_safe_insert(crud, table, df, columns=None, batch_size=100, schema_check=True):
    """
    Guaranteed data insertion with progressive data type relaxation.
    This function will insert ALL rows, modifying problematic data if necessary.

    Args:
        crud: Database CRUD handler
        table (str): Table name
        df (pandas.DataFrame): DataFrame to insert
        columns (list, optional): Column names
        batch_size (int): Size of each batch
        schema_check (bool): Whether to check and adapt to schema

    Returns:
        tuple: (Success count, list of modifications made)
    """
    from tqdm import tqdm
    from datetime import datetime

    logger = logging.getLogger(__name__)

    if columns is None:
        columns = list(df.columns)

    # Step 1: Get the table schema and data types
    column_types = {}
    if schema_check:
        try:
            schema_query = f"""
            SELECT
                COLUMN_NAME,
                DATA_TYPE,
                CHARACTER_MAXIMUM_LENGTH,
                NUMERIC_PRECISION,
                NUMERIC_SCALE,
                IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{table}'
            ORDER BY ORDINAL_POSITION
            """
            schema_result = crud.db_client.execute_query(schema_query, fetch_as_dict=True)

            # Create a mapping of column names to their SQL Server types
            for col in schema_result:
                column_types[col['COLUMN_NAME'].lower()] = {
                    'type': col['DATA_TYPE'].lower(),
                    'max_length': col['CHARACTER_MAXIMUM_LENGTH'],
                    'precision': col['NUMERIC_PRECISION'],
                    'scale': col['NUMERIC_SCALE'],
                    'nullable': col['IS_NULLABLE'] == 'YES'
                }

            logger.info(f"Retrieved schema for table {table} with {len(column_types)} columns")
        except Exception as e:
            logger.warning(f"Could not retrieve schema: {str(e)}. Proceeding without schema checks.")

    # Step 2: Make a copy of the DataFrame to avoid modifying the original
    safe_df = df.copy()

    # Step 3: Convert DataFrame to list of tuples
    rows = [tuple(row) for row in df.values]
    total_rows = len(rows)

    # Step 4: Process in batches
    success_count = 0
    modifications = []

    with tqdm(total=total_rows, desc=f"Inserting into {table}") as pbar:
        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            batch = rows[batch_start:batch_end]

            try:
                # First attempt: Try the standard insert
                result = crud.create_without_progress(table, batch, columns)
                if result:
                    success_count += len(batch)
                    pbar.update(len(batch))
                    continue
            except Exception as e:
                logger.warning(f"Batch {batch_start}-{batch_end} failed with standard insert: {str(e)}")

            # If batch insert fails, try row by row
            logger.info(f"Processing batch {batch_start}-{batch_end} row by row")
            for i, row_tuple in enumerate(batch):
                row_index = batch_start + i
                row_inserted = False

                # Multiple fallback attempts for each row
                fallback_levels = [
                    "standard",        # 1. Standard insert attempt
                    "clean",           # 2. Basic cleaning
                    "deep_clean",      # 3. Deep cleaning with non-printable character removal
                    "aggressive",      # 4. Aggressive type conversion
                    "super_safe"       # 5. Ultra-safe fallback (convert everything to strings)
                ]

                for level in fallback_levels:
                    if row_inserted:
                        break

                    try:
                        # Process the row according to the fallback level
                        processed_row = process_row_with_fallback(
                            row_tuple,
                            columns,
                            level,
                            column_types,
                            row_index
                        )

                        # Try to insert the processed row
                        result = crud.create_without_progress(table, [processed_row], columns)
                        if result:
                            success_count += 1
                            pbar.update(1)
                            row_inserted = True
                            if level != "standard":
                                modifications.append({
                                    "row": row_index,
                                    "level": level,
                                    "original": row_tuple,
                                    "modified": processed_row
                                })
                    except Exception as e:
                        if level == "super_safe":
                            logger.error(f"Failed to insert row {row_index} even with super_safe fallback: {str(e)}")
                            # Last resort: Create an alternate version of the row with NULL for all non-string fields
                            try:
                                emergency_row = create_emergency_row(row_tuple, columns, column_types, row_index)
                                result = crud.create_without_progress(table, [emergency_row], columns)
                                if result:
                                    success_count += 1
                                    pbar.update(1)
                                    row_inserted = True
                                    modifications.append({
                                        "row": row_index,
                                        "level": "emergency",
                                        "original": row_tuple,
                                        "modified": emergency_row
                                    })
                            except Exception as final_e:
                                logger.error(f"Emergency insert failed for row {row_index}: {str(final_e)}")
                                # Update progress bar even for failed rows
                                pbar.update(1)
                        else:
                            logger.warning(f"Fallback {level} failed for row {row_index}: {str(e)}")

    # Report results
    logger.info(f"Inserted {success_count}/{total_rows} rows successfully")
    logger.info(f"Made {len(modifications)} modifications to ensure data insertion")

    # If any modifications were made, save them to a log file for reference
    if modifications:
        try:
            import json
            from pathlib import Path

            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            log_file = log_dir / f"data_modifications_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            with open(log_file, 'w') as f:
                json.dump(modifications, f, indent=2, default=str)

            logger.info(f"Saved modification details to {log_file}")
        except Exception as e:
            logger.error(f"Failed to save modification log: {str(e)}")

    return success_count, modifications

def process_row_with_fallback(row_tuple, columns, level, column_types, row_index):
    """
    Process a row with different levels of fallback strategies.

    Args:
        row_tuple (tuple): The original row data
        columns (list): Column names
        level (str): Fallback level
        column_types (dict): Column type information
        row_index (int): Row index for logging

    Returns:
        tuple: Processed row data
    """
    import re
    from datetime import datetime

    # For standard level, return the original row
    if level == "standard":
        return row_tuple

    # Create a list to hold the processed values
    processed_values = list(row_tuple)

    # Process each field according to the fallback level
    for i, value in enumerate(processed_values):
        col_name = columns[i].lower() if i < len(columns) else f"column_{i}"
        col_info = column_types.get(col_name, {})
        col_type = col_info.get('type', 'unknown')

        # Basic cleaning (level 1)
        if level == "clean":
            # Handle basic string cleaning
            if isinstance(value, str):
                # Remove null bytes and some control characters
                processed_values[i] = value.replace('\x00', '').replace('\x1a', '')

                # For date columns, try to standardize format
                if col_type in ('datetime', 'date', 'smalldatetime'):
                    try:
                        # Try to convert to datetime with standard formats
                        for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S'):
                            try:
                                date_obj = datetime.strptime(value, fmt)
                                processed_values[i] = date_obj.strftime('%Y-%m-%d')
                                break
                            except ValueError:
                                continue
                    except Exception:
                        # Keep original if conversion fails
                        pass

        # Deep cleaning (level 2)
        elif level == "deep_clean":
            # Handle more aggressive string cleaning
            if isinstance(value, str):
                # Remove all non-printable and potentially problematic characters
                clean_value = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', value)

                # For numeric columns, remove any non-numeric characters except decimal separator
                if col_type in ('int', 'bigint', 'smallint', 'tinyint', 'decimal', 'numeric', 'float', 'real', 'money'):
                    clean_value = re.sub(r'[^0-9.,\-+]', '', clean_value.replace(',', '.'))
                    try:
                        if clean_value.strip():
                            clean_value = float(clean_value)
                            if clean_value.is_integer() and col_type in ('int', 'bigint', 'smallint', 'tinyint'):
                                clean_value = int(clean_value)
                    except ValueError:
                        # If conversion fails, keep as string
                        pass

                processed_values[i] = clean_value

            # Handle None values
            elif value is None:
                # For non-nullable columns, provide a default
                if col_info.get('nullable', True) == False:
                    if col_type in ('int', 'bigint', 'smallint', 'tinyint'):
                        processed_values[i] = 0
                    elif col_type in ('decimal', 'numeric', 'float', 'real', 'money'):
                        processed_values[i] = 0.0
                    elif col_type in ('datetime', 'date', 'smalldatetime'):
                        processed_values[i] = '1900-01-01'
                    else:
                        processed_values[i] = ''

        # Aggressive type conversion (level 3)
        elif level == "aggressive":
            # Force type conversion based on column type
            if col_type in ('int', 'bigint', 'smallint', 'tinyint'):
                try:
                    if value is None or (isinstance(value, str) and not value.strip()):
                        processed_values[i] = 0
                    else:
                        # For strings, first clean them
                        if isinstance(value, str):
                            value = re.sub(r'[^0-9.,\-+]', '', value.replace(',', '.'))

                        # Then convert to int, handling float values
                        if value == '' or value is None:
                            processed_values[i] = 0
                        else:
                            processed_values[i] = int(float(value))
                except Exception:
                    processed_values[i] = 0

            elif col_type in ('decimal', 'numeric', 'float', 'real', 'money'):
                try:
                    if value is None or (isinstance(value, str) and not value.strip()):
                        processed_values[i] = 0.0
                    else:
                        # For strings, first clean them
                        if isinstance(value, str):
                            value = re.sub(r'[^0-9.,\-+]', '', value.replace(',', '.'))

                        # Then convert to float
                        if value == '' or value is None:
                            processed_values[i] = 0.0
                        else:
                            processed_values[i] = float(value)
                except Exception:
                    processed_values[i] = 0.0

            elif col_type in ('datetime', 'date', 'smalldatetime'):
                try:
                    if value is None or (isinstance(value, str) and not value.strip()):
                        processed_values[i] = '1900-01-01'
                    elif isinstance(value, str):
                        # Try to parse with multiple formats
                        for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S'):
                            try:
                                date_obj = datetime.strptime(value, fmt)
                                processed_values[i] = date_obj.strftime('%Y-%m-%d')
                                break
                            except ValueError:
                                continue
                        else:
                            # If all formats fail, use default
                            processed_values[i] = '1900-01-01'
                except Exception:
                    processed_values[i] = '1900-01-01'

            elif col_type in ('char', 'varchar', 'nchar', 'nvarchar', 'text', 'ntext'):
                try:
                    if value is None:
                        processed_values[i] = ''
                    else:
                        # Convert to string and strip problematic characters
                        str_value = str(value)
                        # Remove all non-printable and potentially problematic characters
                        processed_values[i] = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', str_value)

                        # Limit length if needed
                        max_len = col_info.get('max_length')
                        if max_len and max_len > 0 and len(processed_values[i]) > max_len:
                            processed_values[i] = processed_values[i][:max_len]
                except Exception:
                    processed_values[i] = ''

        # Super safe fallback (level 4) - Convert everything to strings
        elif level == "super_safe":
            try:
                if value is None:
                    if col_info.get('nullable', True):
                        processed_values[i] = None
                    else:
                        # For non-nullable columns
                        if col_type in ('int', 'bigint', 'smallint', 'tinyint'):
                            processed_values[i] = 0
                        elif col_type in ('decimal', 'numeric', 'float', 'real', 'money'):
                            processed_values[i] = 0.0
                        elif col_type in ('datetime', 'date', 'smalldatetime'):
                            processed_values[i] = '1900-01-01'
                        else:
                            processed_values[i] = ''
                else:
                    # For numeric types
                    if col_type in ('int', 'bigint', 'smallint', 'tinyint'):
                        try:
                            processed_values[i] = int(0)  # Default to 0
                        except:
                            processed_values[i] = 0

                    # For decimal types
                    elif col_type in ('decimal', 'numeric', 'float', 'real', 'money'):
                        try:
                            processed_values[i] = float(0.0)  # Default to 0.0
                        except:
                            processed_values[i] = 0.0

                    # For date types
                    elif col_type in ('datetime', 'date', 'smalldatetime'):
                        processed_values[i] = '1900-01-01'  # Default date

                    # For string types
                    else:
                        # Convert everything to a simple ASCII string
                        safe_str = str(value)
                        # Keep only ASCII printable characters
                        safe_str = ''.join(c for c in safe_str if ord(c) >= 32 and ord(c) < 127)

                        # Limit length if needed
                        max_len = col_info.get('max_length')
                        if max_len and max_len > 0 and len(safe_str) > max_len:
                            safe_str = safe_str[:max_len]

                        processed_values[i] = safe_str
            except Exception:
                # Last resort - just use empty or zero values
                if col_type in ('int', 'bigint', 'smallint', 'tinyint'):
                    processed_values[i] = 0
                elif col_type in ('decimal', 'numeric', 'float', 'real', 'money'):
                    processed_values[i] = 0.0
                elif col_type in ('datetime', 'date', 'smalldatetime'):
                    processed_values[i] = '1900-01-01'
                else:
                    processed_values[i] = ''

    return tuple(processed_values)

def create_emergency_row(row_tuple, columns, column_types, row_index):
    """
    Create an emergency version of the row with defaults for all potentially problematic columns.

    Args:
        row_tuple (tuple): The original row data
        columns (list): Column names
        column_types (dict): Column type information
        row_index (int): Row index for logging

    Returns:
        tuple: Emergency row with safe values
    """
    logger = logging.getLogger(__name__)

    logger.warning(f"Creating emergency row for row {row_index}")

    # Create a list to hold the emergency values
    emergency_values = []

    # Process each field
     # Process each field
    for i, value in enumerate(row_tuple):
        col_name = columns[i].lower() if i < len(columns) else f"column_{i}"
        col_info = column_types.get(col_name, {})
        col_type = col_info.get('type', 'unknown')

        # Use safe default values for each type
        if col_type in ('int', 'bigint', 'smallint', 'tinyint'):
            emergency_values.append(0)
        elif col_type in ('decimal', 'numeric', 'float', 'real', 'money'):
            emergency_values.append(0.0)
        elif col_type in ('datetime', 'date', 'smalldatetime'):
            emergency_values.append('1900-01-01')
        elif col_type in ('bit', 'boolean'):
            emergency_values.append(0)
        else:
            # For string types, preserve original if possible but with extreme cleaning
            if isinstance(value, str):
                # Keep only basic ASCII characters
                safe_str = ''.join(c for c in value if ord(c) >= 32 and ord(c) < 127)

                # Limit length if needed
                max_len = col_info.get('max_length')
                if max_len and max_len > 0 and len(safe_str) > max_len:
                    safe_str = safe_str[:max_len]

                emergency_values.append(safe_str)
            else:
                emergency_values.append('')

    return tuple(emergency_values)

def fix_encoding_issues(df):
    """
    Fix potential encoding issues in string columns in the DataFrame.

    This function addresses common encoding problems that can occur when data is
    extracted from systems like SAP and tries to normalize all text data.

    Args:
        df (pandas.DataFrame): The DataFrame to process

    Returns:
        pandas.DataFrame: DataFrame with fixed encoding in string columns
    """
    # Make a copy to avoid modifying the original
    fixed_df = df.copy()

    # Process each string column
    for col in fixed_df.select_dtypes(include=['object']).columns:
        # Apply a series of fixes to each string value
        fixed_df[col] = fixed_df[col].apply(lambda x: fix_string_encoding(x) if isinstance(x, str) else x)
        logger.debug(f"Fixed encoding issues in column: {col}")

    return fixed_df

def fix_string_encoding(text):
    """
    Apply multiple encoding fixes to a string.

    Args:
        text (str): The string to fix

    Returns:
        str: The fixed string
    """
    if not isinstance(text, str):
        return text

    # Step 1: Try to fix common encoding issues
    try:
        # Fix double-encoded UTF-8 (a common issue)
        try:
            # Check if this might be double-encoded UTF-8
            if any(ord(c) > 127 for c in text):
                # Try to decode as latin1 and then re-encode as utf-8
                text = text.encode('latin1').decode('utf-8')
        except Exception:
            pass

        # Step 2: Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)

        # Step 3: Normalize Unicode (convert combined characters to their canonical form)
        text = unicodedata.normalize('NFC', text)

        # Step 4: Replace any remaining invalid characters with a replacement character
        # This helps with SQL Server compatibility
        text = ''.join(c if ord(c) < 55296 or ord(c) > 57343 else '?' for c in text)

    except Exception:
        # If any error occurs, fall back to a simpler approach
        # Remove any non-ASCII characters
        text = ''.join(c for c in text if ord(c) < 128)

    return text

def process_all_data(df, dmkbi_crud, table_name, columns=None):
    """
    Process and insert all data using the ultra-safe approach

    Args:
        df (pandas.DataFrame): The DataFrame to process
        dmkbi_crud: The CRUD handler for database operations
        table_name (str): The name of the target table
        columns (list, optional): List of column names

    Returns:
        bool: True if successful
    """
    # Step 1: Fix any encoding issues in the data
    #logger.info("Fixing encoding issues in the data...")
    fixed_df = fix_encoding_issues(df)

    # Step 2: Apply the ultra-safe insertion process
    logger.info(f"Inserting {len(fixed_df)} rows with ultra-safe insertion process...")
    success_count, modifications = ultra_safe_insert(
        dmkbi_crud,
        table_name,
        fixed_df,
        columns=columns or list(fixed_df.columns),
        batch_size=100
    )

    # Step 3: Report results
    success_rate = (success_count / len(fixed_df)) * 100
    logger.info(f"Inserted {success_count} out of {len(fixed_df)} rows ({success_rate:.2f}%)")
    logger.info(f"Made {len(modifications)} modifications to ensure data insertion")

    return success_count == len(fixed_df)

def delete_existing_data(crud, table):
    """
    Delete existing data from the table.

    Args:
        crud: Database CRUD handler
        table (str): Table name

    Returns:
        bool: True if deletion was successful
    """
    try:

        # check table exist
        if not crud.table_exists(table):
            logger.warning(f"Table '{table}' does not exist.")
            return False

        delete_query = f"DELETE FROM {table}"
        result = crud.execute_raw_query(delete_query)
        if result is not None:
            logger.info(f"Deleted existing data from {table}")
            return True
        else:
            logger.warning(f"Deletion query returned no result for {table}")
            return False
    except Exception as e:
        logger.error(f"Error deleting data from {table}: {str(e)}")
        return False

def clean_files_in_directory(directory, pattern="*.txt"):
    """
    Clean up files in the specified directory matching the given pattern.

    Args:
        directory (str): Directory path
        pattern (str): File pattern to match (default: "*.txt")

    Returns:
        int: Number of files deleted
    """
    import os
    import glob

    files = glob.glob(os.path.join(directory, pattern))
    deleted_count = 0

    for file in files:
        try:
            os.remove(file)
            deleted_count += 1
            logger.info(f"Deleted file: {file}")
        except Exception as e:
            logger.error(f"Error deleting file {file}: {str(e)}")

    logger.info(f"Deleted {deleted_count} files from {directory} matching pattern {pattern}")
    return deleted_count
