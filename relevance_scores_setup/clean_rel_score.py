import pandas as pd
import numpy as np

# Define possible values
possible_values = [0, 1, 2, 3, 4, 5]

def clean_rel_score(value):
    try:
        # Remove spaces and check each part
        for part in str(value).replace(" ", "").split():
            if part in map(str, possible_values):
                return int(part)
    except Exception as e:
        print(f"Error processing value '{value}': {e}")
    return None

# Read CSV file
df = pd.read_csv('bpwithbm25withrelscore.csv')

# Calculate mean and quantiles
mean_score = df['BM25score'].mean()
quantiles = np.quantile(df['BM25score'].dropna(), [i / len(possible_values) for i in range(len(possible_values))])

def assign_cleaned_score(bm25_score):
    for i, q in enumerate(quantiles):
        if bm25_score <= q:
            return possible_values[i]
    return int(possible_values[-1])

# Clean the rel_score and assign cleaned scores
df['rel_score'] = df['rel_score'].apply(clean_rel_score)
df.loc[df['rel_score'].isnull(), 'rel_score'] = df['BM25score'].apply(assign_cleaned_score)

# Drop rows where 'query' is empty
df = df[df['query'].notna()]

# Write to a new CSV file
df.to_csv('bpwithbm25withrelscore_cleaned.csv', index=False)
