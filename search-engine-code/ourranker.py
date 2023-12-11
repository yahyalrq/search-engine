import json
import gzip
import csv
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from vector_ranker import VectorRanker
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from document_preprocessor import RegexTokenizer
from collections import Counter, defaultdict
from indexing import IndexType, Indexer
from ranker import Ranker, BM25,CrossEncoderScorer
from network_features import NetworkFeatures
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from l2r import LambdaMART, L2RFeatureExtractor, L2RRanker
import shelve
import os
from indexing import IndexType, InvertedIndex, BasicInvertedIndex, Indexer
import orjson
import pickle
from dotenv import load_dotenv
import os
load_dotenv()


mongo_key=os.getenv("Mongo_KEY")

class OurRanker:
    def __init__(self) -> None:
        self.preprocessor = RegexTokenizer('\w+')
        self.stopwords = set()
        self.doc_category_info = dict()
        with open('../stopwords.txt', 'r', encoding='utf-8') as file:
            for stopword in file:
                self.stopwords.add(stopword)
        
        try:
            print("Loading indexes")
            self.main_index=self.create_new_index()
            self.title_index=self.create_new_index()
            self.main_index.load(mongo_key,  "Processed_Data", "MainIndex")
            self.title_index.load(mongo_key,  "Processed_Data", "TitleIndex")

        except:
            print("Creating index because loading failed")
            self.main_index=Indexer.create_index(IndexType.InvertedIndex, mongo_key,
                        "Processed_Data", "processed_books",
                        RegexTokenizer('\\w+'), set("../stopwords.txt"),
                        0, text_key="description")

            self.main_index.save(mongo_key,  "Processed_Data", "MainIndex")

            self.title_index=Indexer.create_index(IndexType.InvertedIndex, mongo_key,
                    "Processed_Data", "processed_books",
                    RegexTokenizer('\\w+'), set("../stopwords.txt"),
                    0, text_key="title")
            
            self.title_index.save(mongo_key,  "Processed_Data", "TitleIndex")

        self.raw_text_dict, self.raw_title_dict=Indexer.build_raw_textandtitle_dict(mongo_key,"Processed_Data","processed_books")

    def create_new_index(self) -> InvertedIndex:
        index = BasicInvertedIndex()
        return index 
    

    def query_with_l2r_with_BM25(self):

        self.ce_scorer = CrossEncoderScorer(self.raw_text_dict)

        self.recognized_categories = set(['Category 100', 'Category 11'])

        self.ranker = Ranker(self.main_index, self.preprocessor,self.stopwords, BM25(self.main_index), self.raw_text_dict)

        self.fe = L2RFeatureExtractor(self.main_index,
                                      self.title_index,
                                      self.doc_category_info,
                                      self.preprocessor,
                                      self.stopwords,
                                      self.recognized_categories,
                                      self.ce_scorer)
    
        l2r = L2RRanker(self.main_index, self.title_index, self.preprocessor,
                        self.stopwords, self.ranker, self.fe,self.raw_text_dict, self.raw_title_dict)
    
        l2r.train('../relevancescorestrain.csv')
        doc_rankings = {}
        query="Best Books"
        print("IN QUERY", query)
        ranked_docs = l2r.query(query)
        print("Finished querying", query)
        doc_ids = [doc['doc_id'] for doc in ranked_docs]
        doc_rankings[query] = doc_ids
        with open('doc_rankings.json', 'w') as json_file:
            json.dump(doc_rankings, json_file, indent=4)
        return doc_rankings
    
    def query_with_BM25(self):

        self.ranker = Ranker(self.main_index, self.preprocessor,self.stopwords, BM25(self.main_index), self.raw_text_dict)
        doc_rankings = {}
        query="best"
        print("IN QUERY", query)
        ranked_docs = self.ranker.query(query)
        print("Finished querying", query)
        doc_ids = [doc['doc_id'] for doc in ranked_docs]
        doc_rankings[query] = doc_ids
        with open('doc_rankings.json', 'w') as json_file:
            json.dump(doc_rankings, json_file, indent=4)
        return doc_rankings
    
    
problem=OurRanker()
#problem.query_with_l2r_with_BM25()
problem.query_with_BM25()
