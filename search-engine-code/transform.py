import json
import os

def read_jsonl_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)

def convert_to_goal_format(input_file, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for user_data in read_jsonl_file(input_file):
        user_id = user_data["user_id"]
        seed_docs = user_data["seed_docs"]

        # Create a new file for each user
        with open(f"{output_dir}/user_{user_id}_dataset.jsonl", 'w') as output_file:
            for doc in seed_docs:
                # Assuming you want to keep the docid, title, text, and categories fields as is
                output_json = {
                    "docid": doc["docid"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "categories": doc["categories"]
                }
                output_file.write(json.dumps(output_json) + '\n')

# Usage
input_file_path = '../personalization.jsonl' # Replace with your file path
output_directory = 'datasets' # Replace with your desired output directory
convert_to_goal_format(input_file_path, output_directory)
