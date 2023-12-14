import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ndcg_score, average_precision_score
import numpy as np
from SearchEngine import SearchEngine
from pymongo import MongoClient
import math

def map_score(search_result_relevances: list[int], cut_off=10) -> float:
    if not search_result_relevances:
        return 0.0

    cumulative_precisions = []
    relevant_count = 0

    total_relevant = sum(1 for score in search_result_relevances if score == 1)

    if total_relevant == 0:
        return 0.0

    for idx, is_relevant in enumerate(search_result_relevances[:cut_off], 1):
        if is_relevant:
            relevant_count += 1
            precision_at_rank = relevant_count / idx
            cumulative_precisions.append(precision_at_rank)

    average_precision = sum(cumulative_precisions) / total_relevant
    return average_precision


def ndcg_score(search_result_relevances: list[float], 
            ideal_relevance_score_ordering: list[float], cut_off=10) -> float:
    def dcg(relevances, cut_off):
        dcg_val = relevances[0]
        for idx, rel in enumerate(relevances[1:cut_off], 2):
            dcg_val += rel / math.log2(idx)
        return dcg_val
    
    actual_dcg = dcg(search_result_relevances, cut_off)
    ideal_dcg = dcg(ideal_relevance_score_ordering, cut_off)
    
    if ideal_dcg == 0:
        return 0.0
    
    ndcg_val = actual_dcg / ideal_dcg
    return ndcg_val

df = pd.read_csv("../relevancescoresdev.csv")

docids = df['docid'].tolist()
queries = df['query'].tolist()
relevance_scores = df['rel'].tolist()

query_doc_to_relevance = {(row['query'], row['docid']): row['rel'] for _, row in df.iterrows()}
print("QUERY to doc relevance", query_doc_to_relevance)
print("\n \n")

client = MongoClient("mongodb+srv://yahya:Yahya123@ir-final.8vivaaw.mongodb.net/?retryWrites=true&w=majority")
db = client["Processed_Data"]
collection = db["processed_books"]
docid_to_text = {}
for document in collection.find():
    docid=str(document["_id"])
    text_description = document.get("description", '')
    docid_to_text[docid] = text_description

query_to_docs = {}
for docid, query in zip(docids, queries):
    if docid in docid_to_text:
        query_to_docs.setdefault(query, []).append((docid, docid_to_text[docid]))


ndcg_scores = {}
map_scores = {}


model = SearchEngine()
individual_ndcg_scores = []
individual_map_scores = []

for query, query_relevance in zip(queries, relevance_scores):

    doc_scores=model.query_with_BM25_forplot(query)
    doc_scores = [(doc['doc_id'], doc["score"]) for doc in doc_scores]
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    print(doc_scores[0:5])

    sorted_doc_relevances = [query_doc_to_relevance[(query, doc_id)] for doc_id, _ in doc_scores]

    ndcg = ndcg_score(sorted_doc_relevances, list(query_doc_to_relevance.values()))
    map = map_score(sorted_doc_relevances)
    individual_ndcg_scores.append(ndcg)
    individual_map_scores.append(map)


ndcg_scores["BM25"] = sum(individual_ndcg_scores) / len(individual_ndcg_scores)
map_scores["BM25"] = sum(individual_map_scores) / len(individual_map_scores)

df_ndcg = pd.DataFrame.from_dict(ndcg_scores, orient='index', columns=['NDCG@10'])
df_map = pd.DataFrame.from_dict(map_scores, orient='index', columns=['MAP@10'])
df_scores = pd.concat([df_ndcg, df_map], axis=1).reset_index()

melted_df = pd.melt(df_scores, ['index'])

# Plot NDCG
sns.barplot(x=df_ndcg.index, y=df_ndcg['NDCG@10'])
plt.title('Mean NDCG@10 for each Model')
plt.xlabel('Models ')
plt.ylabel('NDCG@10 Scores')
plt.xticks(rotation=90)
plt.show()

# Plot MAP
sns.barplot(x=df_map.index, y=df_map['MAP@10'])
plt.title('Mean MAP@10 for each Model')
plt.xlabel('Models')
plt.ylabel('MAP@10 Scores')
plt.xticks(rotation=90)
plt.show()
