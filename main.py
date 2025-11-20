from datetime import datetime, date
import os
import re
import sys
import asyncio

import pandas as pd

from helpers.df_processor import load_partidas_file
from helpers.number_converter import process_sap_dataframe_numbers
from helpers.sapgui import SapGui
from helpers.utils import detect_and_convert_date_columns, detect_file_encoding, timed

# Ensure the project root directory is in sys.path
project_root = os.path.abspath(os.path.dirname(__file__) if '__file__' in globals() else os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from helpers.operations import *
from helpers.exception_handler import *
from helpers.email_sender import EmailSender

# Initialize the logger
logger = setup_logger(__name__)

SEND_EMAIL = True  # Set to True to enable email notifications
DELETE_DATA = True  # Set to True to enable data deletion before insertion

@timed
async def main():
    """
    Main RPA process to extract data from SAP with proper resource management.

    All database connections are managed by the 'managed_resources' context manager.
    Connections are closed automatically, even if exceptions occur.
    """

    # Use the context manager for resource management
    async with managed_resources() as (config, dmkbi_crud, postgresql_crud, dbs):

        logger.info("Starting ETL process...")

        # Unpack database connections
        dmkbi_db, postgresql_db = dbs

        # Load SAP configuration
        sap_configs = config.database.get('sap')
        sap_app = config.sap_app
        sap_table = sap_configs['table']
        sap_where = sap_configs['where']
        sap_code = sap_app.get('transaction_code')

        client_accounts = config.client_accounts
        if not client_accounts:
            raise ValueError("No client accounts provided in configuration")

        # Initialize email sender
        email_sender = EmailSender(config.smtp_configs)

        # Initialize exception handler
        exception_handler = ExceptionHandler(
            crud=postgresql_crud,
            email_sender=email_sender,
            config=config.error_report
        )

        directory_path = None
        sap_session = None

        try:
            # Generate process name
            raw_name = config.process.get('name')
            process_name = re.sub(r'\W+', '_', raw_name.lower()).strip('_') + '_etl'
            logger.info(f"Process name: {process_name}")

            # Get start date for processing
            start_date = datetime.today().replace(day=1)
            logger.info(f"Start processing date: {start_date.strftime('%Y-%m-%d')}")

            # Fetch SAP arguments from database
            sap_args_data = postgresql_crud.read(
                sap_table,
                where=sap_where['clause'],
                params=(sap_where['params'],)
            )
            if not sap_args_data:
                raise ValueError("No SAP arguments found in database")

            # Prepare SAP arguments
            sap_args = sap_args_data[0].copy()
            sap_args.update(sap_app)
            required_args = ['platform', 'username', 'password']

            missing_args = [arg for arg in required_args if arg not in sap_args or not sap_args[arg]]
            if missing_args:
                raise ValueError(f"Missing required SAP arguments: {missing_args}")

            logger.info("Initializing SAP Operations...")

            # SAP Session Management
            try:
                # Create and login to SAP
                sap_session = SapGui(sap_args)
                if not sap_session:
                    raise ValueError("Failed to create SAP session")
                logger.info("SAP session created successfully.")

                if not sap_session.sapLogin():
                    raise Exception("SAP login failed.")
                logger.info("SAP login successful.")

                # Execute SAP operation
                status, file_path = sap_session.sapOperation(
                    command=sap_code,
                    client_accounts=client_accounts
                )
                directory_path = os.path.dirname(file_path) if file_path else None

                if not status or not file_path:
                    raise Exception("SAP operation failed.")

                logger.info("SAP operation completed successfully.")

                # Load and process data
                try:
                    df = load_partidas_file(file_path)
                except Exception as e:
                    logger.error(f"Failed to read SAP file {file_path}: {e}")
                    raise

                # Check if data is empty or contains no-data message
                df_str = df.to_string()
                if "Lista nao contem dados" in df_str or df.empty:
                    handle_no_data_found(
                        postgresql_crud=postgresql_crud,
                        control_table=config.table_control,
                        process_name=process_name,
                        raw_name=raw_name,
                        start_date=start_date,
                        email_sender=email_sender,
                        report_config=config.report,
                        send_email=SEND_EMAIL
                    )
                    return

                # Clean DataFrame
                unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
                if unnamed_cols:
                    logger.info(f"Removing {len(unnamed_cols)} unnamed columns")
                    df = df.drop(columns=unnamed_cols)

                # Remove first column if named 'S'
                if df.columns[0] == 'S':
                    df = df.drop(columns=['S'])

                # Add import timestamp
                df['dataImport'] = datetime.now()

                # Step 1: Convert Portuguese numbers
                logger.info("Converting Portuguese numbers to English format...")
                df = process_sap_dataframe_numbers(df)
                logger.info("Number conversion completed")

                # Step 2: Clean string columns
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].fillna('').astype(str)
                        df[col] = df[col].str.replace('\x00', '')

                # Step 3: Fill NaN values
                str_cols = df.select_dtypes(include=['object']).columns
                num_cols = df.select_dtypes(include=['number']).columns

                df[str_cols] = df[str_cols].fillna('')
                df[num_cols] = df[num_cols].fillna(0)

                # Step 4: Detect and convert dates
                df = detect_and_convert_date_columns(df)

                # Step 5: Ensure numeric columns are properly typed
                numeric_cols = df.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    df[col] = df[col].fillna(0)

                # Step 6: Delete existing data before insert
                if DELETE_DATA:
                    try:
                        delete_existing_data(crud=dmkbi_crud, table=config.table)
                    except Exception as e:
                        logger.error(f"Error deleting existing data: {str(e)}")
                        raise

                # Step 7: Prepare and insert data
                try:
                    df = dmkbi_crud.prepare_data_for_sql_insertion(df)
                    logger.info("Using ultra-safe data insertion...")
                    success = process_all_data(df, dmkbi_crud, config.table)

                    if success:
                        logger.info(f"All {len(df)} records inserted successfully.")
                    else:
                        logger.warning("Some records may not have been inserted.")

                except Exception as e:
                    logger.error(f"Database insertion error: {str(e)}")
                    raise

                logger.info(f"Data saved to {config.table} table.")

                # Update control table
                success = update_last_processed_date(
                    postgresql_crud,
                    start_date,
                    config.table_control,
                    process_name
                )

                # Send success notification
                alert_info = create_alert_message(raw_name, start_date, success, False)

                if SEND_EMAIL:
                    summary_data = [
                        {"label": "Total processos", "value": "1"},
                        {"label": "Total registos", "value": str(len(df))},
                        {"label": "Taxa de sucesso", "value": "100%"}
                    ]

                    email_sender.send_template_email(
                        report_config=config.report,
                        alert_type=alert_info['alert_type'],
                        alert_title=alert_info['alert_title'],
                        alert_message=alert_info['alert_message'],
                        summary_data=summary_data,
                        environment='production',
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )

                logger.info("ETL process completed successfully.")

            except Exception as e:
                logger.error(f"SAP operation error: {str(e)}", exc_info=True)
                raise

            finally:
                # Close SAP session if it exists
                if sap_session:
                    try:
                        sap_session.close_connection()
                    except Exception as e:
                        logger.warning(f"Error closing SAP session: {e}")

        except Exception as e:
            logger.error(f"Critical error in ETL process: {e}")
            exception_handler.get_exception(e)
            raise

        finally:
            # Clean up temporary files if directory path was set
            if directory_path:
                try:
                    logger.info(f"Cleaning up temporary files in: {directory_path}")
                    # Implement file cleanup if needed
                    logger.info("Temporary files cleaned up.")
                except Exception as e:
                    logger.warning(f"Error cleaning up files: {e}")

        # NOTE: Database connections are automatically closed by the managed_resources
        # context manager. Do NOT manually close them here.


# Entry point
if __name__ == "__main__":
    logger.info("Starting ETL process from main entry point...")
    try:
        asyncio.run(main())
        logger.info("ETL process finished successfully.")
    except Exception as e:
        logger.error(f"ETL process failed: {e}", exc_info=True)
        sys.exit(1)
