import pandas as pd


df = pd.read_csv('bpwithbm25.csv')

filtered_df = df.groupby('query').apply(lambda x: x.sort_values('BM25score', ascending=False).head(50))


filtered_df = filtered_df.reset_index(drop=True)

filtered_df.to_csv('bpwithbm25.csv', index=False)
