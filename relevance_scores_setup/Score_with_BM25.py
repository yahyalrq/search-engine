import pandas as pd
from rank_bm25 import BM25Okapi
from typing import List
import numpy as np
import re
import json


books_df = pd.read_csv("Books.csv")
books_data = books_df['description'].tolist()
doc_ids = [str(doc_id) for doc_id in books_df['_id'].tolist()]

queries = [
    "what are the best books in the novels genre",
    "how to improve my productivity",
    "Learn everything about my body",
    "science and nature for beginners",
    "any must-read mermaid books you know of?",
    "best books for the crime genre",
    "mixed martial arts techniques",
    "love adive in relationships",
    "who are the most famous opera artists",
    "books to learn dutch",
    "all you need to know about hinduism",
    "kids stories with moral",
    "most famous romance movies",
    "Does god exist",
    "best mixed martial arts books",
    "any must-read trivia books you know of?",
    "looking for books about Scandinavian literature",
    "books about rabbits?",
    "high fantasy books",
    "the European history",
    "English literature books",
    "social work book recommendations",
    "top Judaism books",
    "interested in post-apocalyptic fiction",
    "looking for top race-related books",
    "crime fiction stories",
    "romance books to read",
    "Satanism books",
    "secret hidden books",
    "non censored books",
    "essential readings in the English literature ",
    "notable works in the social work field",
    "celebrated works in crime fiction for an immersive experience.",
    "popular books among LGBTQ+ romance readers.",
    "celebrated works in Judaism for an immersive experience.",
    "notable works in the Satanism field.",
    "leading titles in the post-apocalyptic genre.",
    "books in the race genre that are highly acclaimed.",
        "top choices in classic literature for avid readers",
    "effective strategies to enhance mental well-being",
    "comprehensive guide to understanding human anatomy",
    "introductory resources for environmental science enthusiasts",
    "recommendations for enchanting books about unicorns",
    "essential reads for mystery and detective fiction lovers",
    "key guides for learning Brazilian jiu-jitsu techniques",
    "insights and tips for nurturing healthy relationships",
    "discovering the legends of classical music composers",
    "language learning: Best resources to master German",
    "in-depth exploration of Buddhism fundamentals",
    "charming children's tales with life lessons",
    "iconic films in the historical drama genre",
    "philosophical perspectives on the existence of higher powers",
    "top-rated books for kickboxing enthusiasts",
    "engaging and informative general knowledge books",
    "exploring the richness of Russian literature",
    "delightful reads about the world of cats",
    "must-reads in the realm of epic fantasy",
    "deep dives into ancient Roman history",
    "celebrated works in American literature",
    "valuable reads for aspiring social workers",
    "essential books for understanding Buddhism",
    "immersive dystopian novels for thrill-seekers",
    "insightful books addressing racial issues",
    "captivating novels in the detective fiction genre",
    "heartwarming and memorable romance novels",
    "exploring books on the philosophy of existentialism",
    "uncovering hidden gems in modern literature",
    "books with bold and unfiltered narratives",
    "masterpieces of British literature",
    "impactful publications in the psychology field",
    "gripping reads in the thriller and suspense genre",
    "favored books in the transgender narrative",
    "profound books delving into Islamic teachings",
    "exploratory books on existential philosophy",
    "leading dystopian novels for a thought-provoking read",
    "critically acclaimed works exploring racial identity",
    "what are the best books in the science fiction genre",
    "how to improve my mental health",
    "Learn everything about plant biology",
    "mathematics for beginners",
    "any must-read dragon-themed books you know of?",
    "best books for the historical fiction genre",
    "advice for maintaining friendships",
    "who are the most renowned modern artists",
    "books to learn Spanish",
    "all you need to know about Taoism",
    "children's books about adventure",
    "most famous action movies",
    "Philosophical debates on free will",
    "best books about yoga",
    "any must-read books on ancient civilizations you know of?",
    "looking for books about African literature",
    "books about ocean life?",
    "dystopian novels recommendations",
    "the history of the Mediterranean region",
    "classic American literature books",
    "psychology book recommendations",
    "top books about Christianity",
    "interested in time-travel fiction",
    "looking for top books about cultural diversity",
    "mystery novels suggestions",
    "feel-good books to read",
    "books on atheism",
    "books with unexpected plot twists",
    "uncut and unedited books",
    "essential readings in modern philosophy",
    "notable works in the field of anthropology",
    "celebrated works in mystery fiction for an immersive experience.",
    "popular books among young adult readers.",
    "celebrated works in Christian theology for an immersive experience.",
    "notable works in the field of atheism.",
    "leading titles in the sci-fi genre.",
    "books in the cultural studies genre that are highly acclaimed.",
    "top choices in world literature for avid readers",
    "effective strategies to enhance physical fitness",
    "comprehensive guide to understanding animal behavior",
    "introductory resources for astronomy enthusiasts"
]

def tokenize(text: str) -> List[str]:
    text = str(text)
    return re.findall(r'\w+', text.lower())

def calculate_bm25_scores(query: str, documents: List[str]) -> List[float]:
    tokenized_query = tokenize(query)
    tokenized_docs = [tokenize(doc) for doc in documents]

    bm25 = BM25Okapi(tokenized_docs)
    return bm25.get_scores(tokenized_query)

result_rows = []
for query in queries:
    bm25_scores = calculate_bm25_scores(query, books_data)

    for doc_id, text, score in zip(doc_ids, books_data, bm25_scores):
        result_rows.append({'doc_id': doc_id, 'query': query, 'text': text, 'BM25score': score})


results = pd.DataFrame(result_rows)

# Save the new dataset
results.to_csv('bpwithbm25.csv', index=False)