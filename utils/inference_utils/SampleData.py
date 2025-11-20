import pandas as pd
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the path from the environment variable
inference_sample_data_file_path = os.getenv("INFERENCE_SAMPLE_DATA_FILE_PATH")

# Method to read the file, parse the JSON data, and return the DataFrame
def get_test_dataframe():
    # Define the base path from the environment variable
    file_path = os.path.join(os.path.dirname(__file__), '../../', inference_sample_data_file_path)

    # Read the content of the file
    with open(file_path, 'r') as file:
        raw_test_data = json.load(file)  # Assuming the file contains a valid JSON array

    # Convert the loaded data to a pandas DataFrame
    test_df = pd.DataFrame(raw_test_data)

    # Set pandas display options to show the full content of the 'Question' column
    pd.set_option('display.max_colwidth', None)  # None means no truncation for any column width

    # Return the DataFrame
    return test_df

