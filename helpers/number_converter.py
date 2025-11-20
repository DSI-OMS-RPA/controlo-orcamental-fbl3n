import pandas as pd
import re
import logging
from typing import List, Union, Any

logger = logging.getLogger(__name__)

def convert_portuguese_to_english_number(value: Any) -> Union[float, int, str, None]:
    """
    Convert Portuguese number format to English format.

    Portuguese: 1.234.567,89 (dot as thousand separator, comma as decimal)
    English: 1234567.89 (no thousand separator, dot as decimal)

    Args:
        value: Input value to convert

    Returns:
        Converted numeric value or original if not numeric
    """
    if pd.isna(value) or value is None:
        return None

    if not isinstance(value, str):
        # Already a number, return as-is
        return value

    # Clean whitespace
    value = value.strip()

    if value == '' or value == '-':
        return None

    # ENHANCED PORTUGUESE DETECTION:
    # Pattern 1: Numbers with comma as decimal separator (definitive Portuguese)
    # Examples: 1.234,50 or 1234,50 or 123,45
    if ',' in value and re.match(r'^-?\d{1,3}(?:\.\d{3})*,\d+$', value):
        # Definitely Portuguese format with thousand separators and decimal comma
        converted = value.replace('.', '').replace(',', '.')
        try:
            float_val = float(converted)
            return int(float_val) if float_val.is_integer() else float_val
        except ValueError:
            logger.warning(f"Failed to convert Portuguese number with comma: {value}")
            return value

    # Pattern 2: Numbers with only comma (no dots) - likely Portuguese decimal
    if ',' in value and '.' not in value and re.match(r'^-?\d+,\d+$', value):
        converted = value.replace(',', '.')
        try:
            float_val = float(converted)
            return int(float_val) if float_val.is_integer() else float_val
        except ValueError:
            logger.warning(f"Failed to convert Portuguese decimal: {value}")
            return value

    # Pattern 3: Numbers with dots but no comma - ASSUME Portuguese if it looks like thousand separator
    # This is the tricky case: 3.625 could be 3625 (Portuguese) or 3.625 (English)
    # Heuristic: if it's a "round" number that could be thousands, treat as Portuguese
    if '.' in value and ',' not in value:
        # Check if this could be Portuguese thousands format
        # Examples: 1.000, 2.500, 3.625 (likely Portuguese if >= 1000)
        if re.match(r'^-?\d{1,3}(?:\.\d{3})+$', value):
            # Multiple groups of exactly 3 digits - definitely Portuguese thousands
            converted = value.replace('.', '')
            try:
                return int(converted)
            except ValueError:
                logger.warning(f"Failed to convert Portuguese thousands: {value}")
                return value

        elif re.match(r'^-?\d{1,4}\.\d{3}$', value):
            # Single group: X.XXX format - could be Portuguese thousands
            # Apply heuristic: if the number >= 1000 when dots removed, likely Portuguese
            test_value = value.replace('.', '')
            try:
                numeric_test = int(test_value)
                if numeric_test >= 1000:
                    # Likely Portuguese thousands format
                    logger.info(f"Converting {value} as Portuguese thousands to {numeric_test}")
                    return numeric_test
                else:
                    # Likely English decimal
                    float_val = float(value)
                    return int(float_val) if float_val.is_integer() else float_val
            except ValueError:
                pass

    # Pattern 4: Pure integers or already English format
    if value.replace('-', '').isdigit():
        return int(value)

    # Pattern 5: English format with comma thousand separators: 1,234.56
    if re.match(r'^-?\d{1,3}(?:,\d{3})*\.\d+$', value):
        cleaned = value.replace(',', '')
        try:
            float_val = float(cleaned)
            return int(float_val) if float_val.is_integer() else float_val
        except ValueError:
            pass

    # Pattern 6: Simple English decimal: 123.45 (no thousand separators)
    if re.match(r'^-?\d+\.\d+$', value) and not re.match(r'^-?\d{1,4}\.\d{3}$', value):
        try:
            float_val = float(value)
            return int(float_val) if float_val.is_integer() else float_val
        except ValueError:
            pass

    # If no pattern matches, return original value
    return value

def detect_and_convert_numeric_columns(df: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
    """
    Automatically detect columns with Portuguese numbers and convert them to English format.

    Args:
        df: DataFrame to process
        sample_size: Number of rows to sample for detection

    Returns:
        DataFrame with converted numeric columns
    """
    converted_df = df.copy()
    converted_columns = []

    # Sample data for faster detection
    sample_df = df.head(min(sample_size, len(df)))

    logger.info(f"=== DEBUGGING COLUMN DETECTION ===")
    logger.info(f"Total columns to analyze: {len(df.columns)}")

    for column in df.columns:
        logger.info(f"Analyzing column '{column}':")
        logger.info(f"  Column dtype: {df[column].dtype}")

        # Skip if column is already numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            logger.info(f"  -> Skipping (already numeric)")
            continue

        # Get non-null string values from sample
        sample_values = sample_df[column].dropna()
        if len(sample_values) == 0:
            logger.info(f"  -> Skipping (no sample values)")
            continue

        # Convert to string and check for numeric patterns
        string_values = sample_values.astype(str)

        # Show first 10 sample values
        first_samples = string_values.head(10).tolist()
        logger.info(f"  Sample values: {first_samples}")

        # Count different types of numeric patterns
        portuguese_comma_count = 0  # Numbers with comma decimal
        portuguese_dot_count = 0    # Numbers that look like Portuguese thousands
        english_count = 0           # Clear English format
        numeric_count = 0           # Total numeric-looking values

        for value in string_values:
            value = value.strip()
            if value == '' or value == 'nan':
                continue

            # Portuguese with comma decimal
            if re.match(r'^-?\d{1,3}(?:\.\d{3})*,\d+$|^-?\d+,\d+$', value):
                portuguese_comma_count += 1
                numeric_count += 1
                #logger.debug(f"    Portuguese comma: {value}")
            # Possible Portuguese thousands (X.XXX format where result >= 1000)
            elif re.match(r'^-?\d{1,4}\.\d{3}$', value):
                test_val = value.replace('.', '')
                try:
                    if int(test_val) >= 1000:
                        portuguese_dot_count += 1
                        numeric_count += 1
                        #logger.debug(f"    Portuguese thousands: {value}")
                    else:
                        english_count += 1
                        numeric_count += 1
                        #logger.debug(f"    English decimal: {value}")
                except ValueError:
                    pass
            # Clear English format with comma thousands
            elif re.match(r'^-?\d{1,3}(?:,\d{3})*\.\d+$', value):
                english_count += 1
                numeric_count += 1
                #logger.debug(f"    English thousands: {value}")
            # Simple numbers (integers or simple decimals)
            elif re.match(r'^-?\d+$|^-?\d+\.\d{1,2}$', value):
                numeric_count += 1
                # logger.debug(f"    Simple number: {value}")

        # Decision logic: convert if we have significant numeric content
        threshold = max(1, len(string_values) * 0.2)  # 20% threshold

        portuguese_indicators = portuguese_comma_count + portuguese_dot_count

        logger.info(f"  Analysis results:")
        logger.info(f"    Total samples: {len(string_values)}")
        logger.info(f"    Portuguese comma: {portuguese_comma_count}")
        logger.info(f"    Portuguese dots: {portuguese_dot_count}")
        logger.info(f"    English format: {english_count}")
        logger.info(f"    Total numeric: {numeric_count}")
        logger.info(f"    Threshold needed: {threshold}")
        logger.info(f"    Portuguese indicators: {portuguese_indicators}")

        if numeric_count >= threshold:
            logger.info(f"  -> CONVERTING column '{column}'!")

            # Apply conversion to entire column
            converted_df[column] = df[column].apply(convert_portuguese_to_english_number)
            converted_columns.append(column)

            # Show before/after samples
            before_sample = df[column].head(5).tolist()
            after_sample = converted_df[column].head(5).tolist()
            logger.info(f"    Before: {before_sample}")
            logger.info(f"    After:  {after_sample}")
        else:
            logger.info(f"  -> Skipping (not enough numeric content: {numeric_count} < {threshold})")

    logger.info(f"=== CONVERSION SUMMARY ===")
    if converted_columns:
        logger.info(f"Successfully converted {len(converted_columns)} columns: {converted_columns}")
    else:
        logger.info("No numeric columns detected for conversion")

    return converted_df

def enhanced_clean_numeric_column(value: Any) -> Any:
    """
    Enhanced version of your existing clean_numeric_column function.
    Replaces the original in utils.py
    """
    return convert_portuguese_to_english_number(value)

def force_convert_portuguese_columns(df: pd.DataFrame, column_patterns: List[str] = None) -> pd.DataFrame:
    """
    Force convert specific columns that are known to contain Portuguese numbers.

    Args:
        df: DataFrame to process
        column_patterns: List of column name patterns to force convert

    Returns:
        DataFrame with converted columns
    """
    if column_patterns is None:
        # Common SAP stock column patterns that likely contain Portuguese numbers
        column_patterns = [
            'util', 'livre', 'trans', 'ctr', 'qld', 'restrito', 'bloqueado',
            'devols', 'transito', 'transf', 'blq', 'vas', 'vinc',
            'stock', 'quantidade', 'qtd', 'saldo', 'te'  # Added 'te'
        ]

    converted_df = df.copy()
    converted_columns = []

    logger.info(f"=== FORCE CONVERTING PORTUGUESE COLUMNS ===")

    for column in df.columns:
        # Check if column name matches any pattern
        column_lower = column.lower().replace('.', '').replace(' ', '')
        should_convert = any(pattern in column_lower for pattern in column_patterns)

        if should_convert:  # Remove dtype check - convert regardless of current type
            logger.info(f"Force converting column '{column}' (dtype: {df[column].dtype})")

            # Get sample of non-empty values
            sample_values = df[column].dropna()
            if len(sample_values) > 0:
                before_sample = sample_values.head(5).tolist()
                logger.info(f"  Before: {before_sample}")

                # Apply Portuguese conversion
                converted_df[column] = df[column].apply(convert_portuguese_to_english_number)
                converted_columns.append(column)

                # Show after conversion
                after_sample = converted_df[column].dropna().head(5).tolist()
                logger.info(f"  After:  {after_sample}")
                logger.info(f"  New dtype: {converted_df[column].dtype}")
            else:
                logger.info(f"  Skipping - no values to convert")

    if converted_columns:
        logger.info(f"Force converted {len(converted_columns)} columns: {converted_columns}")
    else:
        logger.info("No columns matched the force conversion patterns")

    return converted_df

# Updated integration function
def process_sap_dataframe_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process SAP DataFrame to handle Portuguese number formatting.
    This function integrates with your existing ETL pipeline.

    Args:
        df: Raw DataFrame from SAP

    Returns:
        DataFrame with properly converted numbers
    """
    logger.info(f"Processing SAP DataFrame with {len(df)} rows and {len(df.columns)} columns")

    # First, try automatic detection and conversion
    converted_df = detect_and_convert_numeric_columns(df)

    # Then, force convert columns that we know should contain Portuguese numbers
    converted_df = force_convert_portuguese_columns(converted_df)

    logger.info("SAP DataFrame number processing completed")
    return converted_df

# Test function to debug your specific data
def test_conversion_with_your_data():
    """Test function using your actual data samples"""
    test_values = [
        "23",       # Should stay 23
        "73",       # Should stay 73
        "49",       # Should stay 49
        "39",       # Should stay 39
        "3.625",    # Portuguese: should become 3625
        "954.706",  # Portuguese: should become 954706
        "9.706,47",  # Portuguese: should become 954706.47
        "128",      # Should stay 128
        "150",      # Should stay 150
        "239"       # Should stay 239
    ]

    print("Testing with your actual data:")
    for test_val in test_values:
        result = convert_portuguese_to_english_number(test_val)
        print(f"'{test_val}' -> {result} ({type(result).__name__})")

if __name__ == "__main__":
    test_conversion_with_your_data()
