'''
A utility script to combine multiple CSV files from the LOBSTER dataset 
into a single CSV file for message and orderbook data. 
'''

import pandas as pd
import glob
import re
import os

# Define file patterns for message and orderbook files and sort them
# Note: MAKE SURE YOU HAVE the raw LOBSTER data files in the specified directory
message_files = sorted(glob.glob("./_data_dwn_43_456__AMZN_2024-08-01_2024-09-01_10/AMZN_*_message_10.csv"))
orderbook_files = sorted(glob.glob("./_data_dwn_43_456__AMZN_2024-08-01_2024-09-01_10/AMZN_*_orderbook_10.csv"))

# date to filter. If no filtering is needed, croppedDate should be empty string and isCropped should be False
isCropped = False
croppedDate = ''

# AD HOC filtering for demonstration purposes
if isCropped:
    message_files = [file for file in message_files if croppedDate in file]
    orderbook_files = [file for file in orderbook_files if croppedDate in file]

# Define column names for message data
message_column_names = [
    'Time', 'EventType', 'OrderID', 'Size', 'Price', 'Direction', 'NotUsed'
]

# Define column names for orderbook data up to level 10
book_column_names = []
for level in range(1, 11):
    book_column_names.extend([f'AskPrice{level}', f'AskSize{level}', f'BidPrice{level}', f'BidSize{level}'])

# Load, clean, and concatenate message data
message_dfs = []
for file in message_files:
    print(f"Processing message file: {file}")
    try:
        # Extract the date from the filename by getting only the basename
        filename = os.path.basename(file)
        date_match = re.search(r"AMZN_(\d{4}-\d{2}-\d{2})_", filename)
        if date_match:
            file_date = date_match.group(1)
            origin_date = pd.Timestamp(f"{file_date} 00:00:00")
        else:
            print(f"Date not found in filename: {filename}")
            continue  # Skip if date is not found

        # Check if file is empty
        if pd.read_csv(file, nrows=1).empty:
            print(f"Skipping empty file: {file}")
            continue

        # Load data, assign columns, and drop unnecessary columns/rows
        data = pd.read_csv(file, header=None, dtype={6: 'object'}, low_memory=False)
        data.columns = message_column_names
        data = data.drop(columns=['NotUsed'])  # Drop 'NotUsed' column immediately
        data = data.drop(0)  # Drop the first row if it's unnecessary

        # Convert 'Time' to datetime using the inferred origin
        data['Time'] = pd.to_datetime(data['Time'], unit='s', origin=origin_date)

        message_dfs.append(data)  # Append to list
    except pd.errors.EmptyDataError:
        print(f"Error: No data in file {file}")
    except Exception as e:
        print(f"An error occurred with file {file}: {e}")

combined_messages = pd.concat(message_dfs, ignore_index=True)

# Ensure Time column is of datetime type
combined_messages['Time'] = pd.to_datetime(combined_messages['Time'])

# Load, clean, and concatenate orderbook data
orderbook_dfs = []
for file in orderbook_files:
    print(f"Processing orderbook file: {file}")
    try:
        # Check if file is empty
        if pd.read_csv(file, nrows=1).empty:
            print(f"Skipping empty file: {file}")
            continue
        
        data = pd.read_csv(file, header=None, low_memory=False)
        data.columns = book_column_names       # Assign orderbook column names
        data = data.drop(0)                    # Drop the first row if it's unnecessary
        orderbook_dfs.append(data)             # Append to list
    except pd.errors.EmptyDataError:
        print(f"Error: No data in file {file}")
    except Exception as e:
        print(f"An error occurred with file {file}: {e}")

combined_orderbooks = pd.concat(orderbook_dfs, ignore_index=True)

# Save the combined dataframes to CSV files
combined_messages.to_csv(f"combined_messages_{croppedDate}.csv", index=False)
combined_orderbooks.to_csv(f"combined_orderbooks_{croppedDate}.csv", index=False)

print("Combined message data shape:", combined_messages.shape)
print("Combined orderbook data shape:", combined_orderbooks.shape)




