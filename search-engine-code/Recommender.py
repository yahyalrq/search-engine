import os
import pymongo
import dotenv
import os
import json
import time
import logging
import pandas as pd
from pymongo import UpdateOne
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
class Recommender:
    def __init__(self, db_name, collection_name) -> None:
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.init_mongo_client()

    def init_mongo_client(self):
        # Initialize MongoDB client
        dotenv.load_dotenv()
        mongodb_user = os.getenv('MongoDBUser')
        mongodb_password = os.getenv('MongoDBPassword')
        mongodb_uri_template = os.getenv('MongoDBURITemplate')
        mongodb_uri = mongodb_uri_template.format(username=mongodb_user, password=mongodb_password)
        self.client = pymongo.MongoClient(mongodb_uri)

    def get_book_vectors(self, book_ids):
        db = self.client[self.db_name]
        collection = db[self.collection_name]
        query = {'doc_id': {'$in': book_ids}}
        book_vectors = collection.find(query)
        return list(book_vectors)
    
    def get_recommendations(self, list_of_book_ids_for_user, list_of_top_100_books):
        # Combine user and top 100 book ids for a single query
        user_books = self.get_book_vectors(list_of_book_ids_for_user)
        search_books = self.get_book_vectors(list_of_top_100_books)

        user_books_features = [book['vector'] for book in user_books]
        search_books_features = [book['vector'] for book in search_books]

        # Convert lists to NumPy arrays for cosine similarity calculation
        user_books_matrix = np.array(user_books_features)
        search_books_matrix = np.array(search_books_features)

        # Calculate cosine similarity
        similarities = cosine_similarity(user_books_matrix, search_books_matrix)

        # Average similarity for each search book
        avg_similarities = np.mean(similarities, axis=0)

        # Sort search books by similarity (and get corresponding book IDs)
        sorted_indices = np.argsort(avg_similarities)[::-1]
        sorted_books = [list_of_top_100_books[idx] for idx in sorted_indices]

        # Return re-ranked list of book IDs
        return sorted_books

    

def get_sample_books():
    # Get sample books
    dotenv.load_dotenv()
    # Get variables from environment
    mongodb_user = os.getenv('MongoDBUser')
    mongodb_password = os.getenv('MongoDBPassword')
    mongodb_uri_template = os.getenv('MongoDBURITemplate')
    # Format the URI with the actual username and password
    mongodb_uri = mongodb_uri_template.format(username=mongodb_user, password=mongodb_password)
    client = pymongo.MongoClient(mongodb_uri)
    db = client["Processed_Data"]
    collection = db["processed_books"]
    # Just retrieve the first 105 book ids
    book_ids = collection.find().limit(105)
    sample_book_ids = []
    for book in book_ids:
        sample_book_ids.append(str(book["_id"]))
    return sample_book_ids
"""
book_ids = get_sample_books()
print(book_ids[:5])
recommender = Recommender("Processed_Data", "Vectorized_books")
user_book_ids = book_ids[:5]
top_100_book_ids = book_ids[5:]
results = recommender.get_recommendations(user_book_ids, top_100_book_ids)
print(results)"""