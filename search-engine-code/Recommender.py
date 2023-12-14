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
        # Fetch book vectors for given book IDs from mongo
        db = self.client[self.db_name]
        collection = db[self.collection_name]
        query = {'doc_id': {'$in': book_ids}}
        book_vectors = collection.find(query)
        return list(book_vectors)
    
    """def get_recommendations(self, list_of_book_ids_for_user, list_of_top_k_books):
        # Combine user and top 100 book ids for a single query
        user_books = self.get_book_vectors(list_of_book_ids_for_user)
        search_books = self.get_book_vectors(list_of_top_k_books)

        user_books_features = [book['vector'] for book in user_books]
        search_books_features = [book['vector'] for book in search_books]


        user_books_matrix = np.array(user_books_features)
        search_books_matrix = np.array(search_books_features)

        similarities = cosine_similarity(user_books_matrix, search_books_matrix)

        avg_similarities = np.mean(similarities, axis=0)

        sorted_indices = np.argsort(avg_similarities)[::-1]
        sorted_books = [list_of_top_k_books[idx] for idx in sorted_indices]

        # Return re-ranked list of book IDs
        return sorted_books"""
    
    def get_recommendations(self, list_of_book_ids_for_user, search_engine_results):
        # Extract book IDs and search engine scores from the tuples
        list_of_top_k_books = []
        search_engine_scores = []
        for book in search_engine_results:
            book_id = book[0]
            score = book[1]
            list_of_top_k_books.append(book_id)
            search_engine_scores.append(score)
        # Fetch book vectors for user books and search engine results
        user_books = self.get_book_vectors(list_of_book_ids_for_user)
        search_books = self.get_book_vectors(list_of_top_k_books)
        # Extract features and calculate cosine similarities
        user_books_features = [book['vector'] for book in user_books]
        search_books_features = [book['vector'] for book in search_books]
        user_books_matrix = np.array(user_books_features)
        search_books_matrix = np.array(search_books_features)
        similarities = cosine_similarity(user_books_matrix, search_books_matrix)
        avg_similarities = np.mean(similarities, axis=0)

        # Normalize cosine similarities and search engine scores
        normalized_cosine_scores = (avg_similarities - np.min(avg_similarities)) / (np.max(avg_similarities) - np.min(avg_similarities))
        normalized_search_scores = (np.array(search_engine_scores) - np.min(search_engine_scores)) / (np.max(search_engine_scores) - np.min(search_engine_scores))

        # Weighted combination of scores
        weight_for_cosine = 0.25  
        weight_for_search = 1 - weight_for_cosine
        combined_scores = weight_for_cosine * normalized_cosine_scores + weight_for_search * normalized_search_scores

        # Sort books based on combined scores
        sorted_indices = np.argsort(combined_scores)[::-1]
        sorted_books = [list_of_top_k_books[idx] for idx in sorted_indices]

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
