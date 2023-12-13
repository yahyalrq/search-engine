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
    def _init_(self) -> None:
        self.book_vectors = None

    def get_book_vectors(self, db_name, collection_name):
        if self.book_vectors == None:
            dotenv.load_dotenv()
            mongodb_user = os.getenv('MongoDBUser')
            mongodb_password = os.getenv('MongoDBPassword')
            mongodb_uri_template = os.getenv('MongoDBURITemplate')
            mongodb_uri = mongodb_uri_template.format(username=mongodb_user, password=mongodb_password)
            client = pymongo.MongoClient(mongodb_uri)
            db = client[db_name]
            collection = db[collection_name]
            book_vectors = collection.find()
            self.book_vectors = book_vectors
            return self.book_vectors
        else:
            return self.book_vectors


    def get_recommendations(self, list_of_book_ids_for_user, list_of_top_100_books):
        """
        Re-rank a list of top search results based on user's book preferences.

        Parameters:
        list_of_book_ids_for_user (list): List of book IDs indicating user preferences.
        list_of_top_100_books (list): List of book IDs from top search results.

        Returns:
        list: Re-ranked list of book IDs from search results.
        """
        # Ensure book vectors are loaded
        if self.book_vectors is None:
            raise ValueError("Book vectors are not loaded.")

        # Convert cursor to list if necessary
        book_vectors = list(self.book_vectors) if isinstance(self.book_vectors, pymongo.cursor.Cursor) else self.book_vectors

        # Extract features for user's books and top 100 books
        user_books_features = [book['vector'] for book in book_vectors if book['book_id'] in list_of_book_ids_for_user]
        search_books_features = [book['vector'] for book in book_vectors if book['book_id'] in list_of_top_100_books]
        search_books_ids = [book['book_id'] for book in book_vectors if book['book_id'] in list_of_top_100_books]

        # Convert lists to NumPy arrays for cosine similarity calculation
        user_books_matrix = np.array(user_books_features)
        search_books_matrix = np.array(search_books_features)

        # Calculate cosine similarity
        similarities = cosine_similarity(user_books_matrix, search_books_matrix)

        # Average similarity for each search book
        avg_similarities = np.mean(similarities, axis=0)

        # Sort search books by similarity (and get corresponding book IDs)
        sorted_indices = np.argsort(avg_similarities)[::-1]
        sorted_books = [search_books_ids[idx] for idx in sorted_indices]

        # Return re-ranked list of book IDs
        return sorted_books