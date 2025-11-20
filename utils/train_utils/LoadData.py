import os
import json
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()


# Method to read the file content, remove variable assignment, and convert to a list
def load_data_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    # Remove the variable name and equals sign to isolate the list
    list_str = re.sub(r'.*\s*=\s*', '', content, count=1).strip()

    try:
        # Use json.loads to parse the content
        return json.loads(list_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []



def get_data():
    train_data_path = os.path.join(os.path.dirname(__file__), '../../', os.getenv("TRAIN_DATA_PATH"))
    test_data_path = os.path.join(os.path.dirname(__file__), '../../', os.getenv("TEST_DATA_PATH"))
    val_data_path = os.path.join(os.path.dirname(__file__), '../../', os.getenv("TRAIN_DATA_PATH"))

    # Read data from the local files
    train_data = load_data_from_file(train_data_path)
    test_data = load_data_from_file(test_data_path)
    val_data = load_data_from_file(val_data_path)

    # Take only first 20 entries
    train_data = train_data[:20]
    test_data = test_data[:20]
    val_data = val_data[-5:]

    # Print the first 3 entries
    # print(f"Last 3 entries for train data: {len(train_data)}")
    # for item in train_data[-3:]:
    #     print(item)

    # Print the last 3 entries
    # print("\nLast 3 entries for test data:")
    # for item in test_data[-3:]:
    #     print(item)

    # Print the last 3 entries
    # print(f"\nLast 3 entries for val data: {len(val_data)}")
    # for item in val_data:
    #     print(item)

    return train_data, test_data,val_data

