import openiap
import logging
from typing import Optional, Dict, Any

# Set logging for debug.
logging.basicConfig(level=logging.INFO)


# Connect and authenticate to the OpenIAP service.
async def connect_to_service() -> Optional[openiap.Client]:
    """
    Returns:
        openiap.Client or None: An authenticated OpenIAP client, or None if there's an error during connection/authentication.
    """
    client = openiap.Client()

    try:
        signin = await client.Signin()
        logging.info(f"Signed in as {signin.name}")
        logging.info(
            "Successfully connected and authenticated with OpenIAP service.")
        return client
    except Exception as e:
        logging.error(
            f"Error connecting or authenticating with OpenIAP service: {e}")
        return None


# Retrieve data from a specified OpenIAP collection based on the provided query.
async def get_data(client: openiap.Client, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Args:
        client (openiap.Client): An authenticated OpenIAP client.
        collection (str): Name of the collection to query from.
        query (dict): The query parameters.

    Returns:
        dict or None: The result of the query or None if an error occurs.
    """
    try:
        result = await client.Query(collectionname=collection, query=query)
        logging.info("Successfully retrieved data.")
        return result
    except Exception as e:
        logging.error(
            f"Error querying OpenIAP collection {collection} with value {query}. Error: {e}")
        return None


# Connect to the OpenIAP service and fetch data.
async def fetch_data(client: openiap.Client, collection, query):
    """
    Returns:
        tuple: A tuple containing the fetched data and the client object.
              If there's an error, the respective element in the tuple will be None.
    """
    try:
        if not client:
            logging.error("Unable to connect to OpenIAP service.")
            return data, client

        data = await get_data(client, collection, query)
        if not data:
            logging.warning("No data retrieved.")
            return None

    except Exception as e:
        logging.error(
            f"An error occurred while connecting or fetching data: {e}")

    return data


# Connect to the OpenIAP service and save data.
async def save_data(client: openiap.Client, collection: str, data: list, field="_id") -> bool:
    """
    Saves data to a specified collection using an existing OpenIAP client.

    Args:
        collection (str): The collection to which the data is to be saved.
        data (list): The data to save.

    Returns:
        bool: True if the data was saved successfully, False otherwise.
    """
    if not collection or not isinstance(data, list) or not data:
        logging.error("Invalid data.")
        return False

    try:
        if not client:
            raise ConnectionError("Failed to connect to OpenIAP service.")

        response = await client.InsertOrUpdateMany(items=data, collectionname=collection, uniqeness=field)
        if response:
            logging.info("Data saved successfully.")
            return True
        else:
            logging.error("Data insertion failed without an exception.")
            return False

    except Exception as e:
        logging.error(
            f"Error saving data to collection {collection}: {e}", exc_info=True)
        return False


# Connect to the OpenIAP service and delete data.
async def delete_data(client: openiap.Client, query: dict, collection: str, recursive: bool = False) -> bool:

    if not collection or not isinstance(query, dict) or not query:
        logging.error("Invalid query.")
        return False

    try:
        if not client:
            raise ConnectionError("Failed to connect to OpenIAP service.")

        response = await client.DeleteMany(query=query, collectionname=collection, recursive=recursive)
        if response:
            logging.info("Data deleted successfully.")
            return True
        else:
            logging.error("Data deletion failed without an exception.")
            return False

    except Exception as e:
        logging.error(
            f"Error deleting data to collection {collection}: {e}", exc_info=True)
        return False
