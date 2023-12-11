import pandas as pd
import openai
from dotenv import load_dotenv
import os
load_dotenv()

df = pd.read_csv('bpwithbm25.csv')

OPENAI_KEY = os.getenv("OPENAI_KEY")

client = openai.Client(api_key=OPENAI_KEY)
valid_scores = {0, 1, 2, 3, 4, 5}


for index, row in df.iterrows():
    query = row['query']
    text = row['text']
    prompt = f"Rate the relevance of the following query basing on the following text on a scale from 0 to 5. The output should be the integer and nothing else other than it. Query: {query}, text: {text}"

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=5
    )


    try:
        rel_score = int(response.choices[0].text.strip())
        if rel_score not in valid_scores:
            raise ValueError("Invalid score")
    except (ValueError, TypeError):
        rel_score = None

    df.loc[index, 'rel_score'] = rel_score

df.to_csv('bpwithbm25withrelscore.csv', index=False)

