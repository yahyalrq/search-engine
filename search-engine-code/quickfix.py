import pandas as pd
from pymongo import MongoClient
import bson

# Read the existing CSV file
df = pd.read_csv("../relevancescoresdev.csv")

# Connect to the MongoDB client
client = MongoClient("mongodb+srv://yahya:Yahya123@ir-final.8vivaaw.mongodb.net/?retryWrites=true&w=majority")
db = client["Processed_Data"]
collection = db["processed_books"]

# Retrieve new document IDs
new_docids = []
for _id in df['docid']:
    object_id = bson.ObjectId(_id)
    print("OBJETC ID", object_id)
    document = collection.find_one({'_id': object_id})  
    print(document)
    if document is not None:  # Check if the document was found
        new_docids.append(str(document["book_id"]))
    else:
        print("what")
        new_docids.append(None)  # Append None or a placeholder if the document is not found
    break
# Replace the 'docid' column with the new IDs
df['docid'] = new_docids

# Save the updated DataFrame to a CSV file
df.to_csv("../new_relevancescores.csv", index=False)
