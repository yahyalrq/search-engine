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
from VectorRepresenter import  VectorBuilder, VectorFeatureExtractor
import shelve
import os
from indexing import IndexType, InvertedIndex, BasicInvertedIndex, Indexer
import orjson
import pickle
from dotenv import load_dotenv
import os
load_dotenv()


mongo_key=os.getenv("Mongo_KEY")

class VectorRunner:
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
            self.author_index=self.create_new_index()
            self.main_index.load(mongo_key,  "Processed_Data", "MainIndex")
            self.title_index.load(mongo_key,  "Processed_Data", "TitleIndex")
            self.author_index.load(mongo_key,  "Processed_Data", "AuthorIndex")

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

            self.author_index=Indexer.create_index(IndexType.InvertedIndex, mongo_key,
                    "Processed_Data", "processed_books",
                    RegexTokenizer('\\w+'), set("../stopwords.txt"),
                    0, text_key="author")
            
            self.author_index.save(mongo_key,  "Processed_Data", "AuthorIndex")

        self.raw_text_dict, self.raw_title_dict,  self.doc_category_info=Indexer.build_raw_textandtitle_dict(mongo_key,"Processed_Data","processed_books")

    def create_new_index(self) -> InvertedIndex:
        index = BasicInvertedIndex()
        return index 
    

    def buildvectors(self):

        self.ce_scorer = CrossEncoderScorer(self.raw_text_dict)

        self.recognized_categories = set([])

        self.fe = VectorFeatureExtractor(self.main_index,
                                      self.title_index,
                                      self.doc_category_info,
                                      self.preprocessor,
                                      self.stopwords,
                                      self.recognized_categories,
                                      self.ce_scorer)
    
        vectorbuilder = VectorBuilder(self.preprocessor,self.fe,self.raw_text_dict, self.raw_title_dict,self.stopwords)
        vectors=vectorbuilder.build_vector_representation()
        self.vector=self.create_new_index()
        self.vector.save_vectorized_features(mongo_key,  "Processed_Data", "Vectorized_books",vectors)
        return
    
    
problem=VectorRunner()
problem.buildvectors()
