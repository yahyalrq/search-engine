from enum import Enum
import json
import os
from tqdm import tqdm
from collections import Counter, defaultdict
import shelve
from document_preprocessor import RegexTokenizer
import gzip
from bisect import insort
import orjson
from dotenv import load_dotenv
import os
from bson import ObjectId
load_dotenv()
from pymongo import MongoClient


class IndexType(Enum):
    InvertedIndex = 'BasicInvertedIndex'
    PositionalIndex = 'PositionalIndex'
    OnDiskInvertedIndex = 'OnDiskInvertedIndex'
    SampleIndex = 'SampleIndex'

class InvertedIndex:
    def __init__(self) -> None:
        self.statistics = defaultdict(Counter)

class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        self.statistics['docmap'] = {}
        self.index = defaultdict(list)
        self.doc_id = 0
  
    def remove_doc(self, docid: int) -> None:
        terms_to_delete = []
        for term in list(self.index.keys()):  
            self.index[term] = [post for post in self.index[term] if post[0] != docid]
            if not self.index[term]:
                terms_to_delete.append(term)

        for term in terms_to_delete:
            del self.index[term]

    def add_doc(self, docid: int, tokens: list[str], genres: list[str]) -> None:
        token_counts = Counter(tokens)

        for token, freq in token_counts.items():
            posting = (docid, freq)
            if token not in self.index:
                self.index[token] = [posting]
            else:
                insort(self.index[token], posting)

        self.statistics['docmap'][docid] = {
            'total_tokens': len(tokens),
            'unique_tokens': len(token_counts),
            'genres': genres  
        }

        return (tokens, dict(token_counts))

    def get_postings(self, term: str) -> list[tuple[int, int]]:
        return self.index.get(term, [])

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        return self.statistics['docmap'].get(doc_id, {})

    def get_term_metadata(self, term: str) -> dict[str, int]:
        frequency = []
        for doc_id, freq in self.index.get(term, []):
            frequency.append(freq)
        return {term: sum(frequency)}

    def get_statistics(self) -> dict:
        filtered_docmap = {k: v for k, v in self.statistics['docmap'].items() if k is not None}
        filtered_index = {k: v for k, v in self.index.items() if k is not None}

        total_docs = len(filtered_docmap)
        total_tokens = sum(meta["total_tokens"] for meta in filtered_docmap.values())
        unique_tokens = len(filtered_index)

        all_genres = set()
        for meta in filtered_docmap.values():
            genres = meta.get("genres", [])
            all_genres.update(genres)

        return {
            "unique_token_count": unique_tokens,
            "total_token_count": total_tokens,
            "stored_total_token_count": sum(len(postings) for postings in filtered_index.values()),
            "number_of_documents": total_docs,
            "mean_document_length": total_tokens / total_docs if total_docs else 0,
            "all_genres": list(all_genres)
        }


    def save(self, mongo_uri, db_name, collection_name) -> None:
            client = MongoClient(mongo_uri)
            db = client[db_name]
            collection = db[collection_name]

            collection.delete_many({})

            for term, postings in self.index.items():
                collection.insert_one({"term": term, "postings": postings})

            collection.insert_one({"type": "statistics", "data": self.statistics})

            print("Index saved to MongoDB.")

    def save_vectorized_features(self, mongo_uri, db_name, collection_name, vectors) -> None:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        collection.delete_many({})

        for doc_id, vector in vectors.items():
            collection.insert_one({"doc_id": doc_id, "vector": vector})

        print("Vector saved to MongoDB.")

    def load(self, mongo_uri, db_name, collection_name) -> None:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        # Check if the collection is empty
        if collection.count_documents({}) == 0:
            raise ValueError(f"The collection '{collection_name}' in database '{db_name}' is empty.")

        self.index = {}
        self.statistics = defaultdict(Counter)

        for document in collection.find({"postings": {"$exists": True}}):
            term = document["term"]
            postings = document["postings"]
            self.index[term] = postings

        stats_doc = collection.find_one({"type": "statistics"})
        if stats_doc:
            self.statistics = stats_doc["data"]

        print("Index loaded from MongoDB.")
class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self, index_name) -> None:
        super().__init__(index_name)
        self.statistics['index_type'] = 'PositionalInvertedIndex'

class OnDiskInvertedIndex(BasicInvertedIndex):
    def __init__(self, shelve_filename) -> None:
        super().__init__()
        self.shelve_filename = shelve_filename
        self.statistics['index_type'] = 'OnDiskInvertedIndex'



class Indexer:
    @staticmethod
    def build_raw_textandtitle_dict(mongo_connection_string: str, database_name: str, 
                            collection_name: str, text_key="description", title_key="title") -> dict:
        client = MongoClient(mongo_connection_string)
        db = client[database_name]
        collection = db[collection_name]
        title_text_dict={}
        raw_text_dict = {}
        doc_category_info={}
        for document in collection.find():
            docid=str(document["_id"])
            text_description = document.get(text_key, '')
            text_title = document.get(title_key, '')
            raw_text_dict[docid] = text_description
            title_text_dict[docid] = text_title
            doc_category_info[docid]=document.get("genres","")
        return raw_text_dict, title_text_dict, doc_category_info
    
    @staticmethod
    def create_index(index_type: IndexType, mongo_connection_string: str,
                     database_name: str, collection_name: str,
                     document_preprocessor: RegexTokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="description",
                     max_docs: int = -1) -> InvertedIndex:
        client = MongoClient(mongo_connection_string)
        db = client[database_name]
        collection = db[collection_name]

        if index_type == IndexType.InvertedIndex:
            index = BasicInvertedIndex()
        else:
            raise ValueError("Unsupported index type")

        freqs = Counter()
        for document in collection.find():
            tokens = document_preprocessor.tokenize(document.get(text_key, ''))
            freqs.update(tokens)

        stop_lower = {word.lower() for word in stopwords}
        valid_tokens = set()
        for token, count in freqs.items():
            if count >= minimum_word_frequency and token.lower() not in stop_lower:
                valid_tokens.add(token)

        doc_count = 0
        for document in collection.find():
            tokens = document_preprocessor.tokenize(document.get(text_key, ''))
            token_set = [t if t in valid_tokens else None for t in tokens]
            docid=str(document["_id"])
            index.add_doc(docid, token_set, document["genres"])
            doc_count += 1
            if max_docs > 0 and doc_count >= max_docs:
                break
        return index
        
    @staticmethod
    def create_index_tempuser(index_type: IndexType, mongo_connection_string: str, 
                            book_user_ids: list, database_name: str, collection_name: str,
                            document_preprocessor: RegexTokenizer, stopwords: set[str],
                            minimum_word_frequency: int, text_key="description",
                            max_docs: int = -1) -> InvertedIndex:
        client = MongoClient(mongo_connection_string)
        db = client[database_name]
        collection = db[collection_name]

        if index_type == IndexType.InvertedIndex:
            index = BasicInvertedIndex()
        else:
            raise ValueError("Unsupported index type")
        
        object_ids = [ObjectId(id) for id in book_user_ids]

        query_filter = {'_id': {'$in': object_ids}}

        freqs = Counter()
        for document in collection.find(query_filter):
            tokens = document_preprocessor.tokenize(document.get(text_key, ''))
            freqs.update(tokens)

        stop_lower = {word.lower() for word in stopwords}
        valid_tokens = set()
        for token, count in freqs.items():
            if count >= minimum_word_frequency and token.lower() not in stop_lower:
                valid_tokens.add(token)

        doc_count = 0
        for document in collection.find(query_filter):
            tokens = document_preprocessor.tokenize(document.get(text_key, ''))
            token_set = [t if t in valid_tokens else None for t in tokens]
            docid = str(document["_id"])
            index.add_doc(docid, token_set, document["genres"])
            doc_count += 1
            if max_docs > 0 and doc_count >= max_docs:
                break

        return index

