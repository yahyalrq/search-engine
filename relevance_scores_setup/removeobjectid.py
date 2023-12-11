import pandas as pd

# Load the CSV file
file_path = 'bpwithbm25withrelscore_cleaned.csv'
data = pd.read_csv(file_path)

# Function to convert ObjectId string to regular string
def convert_objectid_to_string(objectid_str):
    return objectid_str.replace('ObjectId(', '').replace(')', '')

# Apply the function to the 'doc_id' column
data['doc_id'] = data['doc_id'].apply(convert_objectid_to_string)

# Save the updated dataframe to a new CSV file
data.to_csv('relevancescorestrain.csv', index=False)
