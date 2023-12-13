from tqdm import tqdm
import pandas as pd
import lightgbm
from indexing import InvertedIndex, Indexer, IndexType
import multiprocessing
from collections import defaultdict, Counter
import numpy as np
from document_preprocessor import RegexTokenizer
from ranker import Ranker, TF_IDF, BM25, PivotedNormalization, CrossEncoderScorer
import math
from typing import Dict, List, Union, Set
import torch
from transformers import RobertaModel, RobertaTokenizer

class VectorBuilder:
    def __init__(self, document_preprocessor: RegexTokenizer, feature_extractor: 'L2RFeatureExtractor', raw_text_dict=None, raw_title_dict=None, stopwords=None ) -> None:
        self.document_preprocessor = document_preprocessor
        self.feature_extractor = feature_extractor
        self.raw_text_dict = raw_text_dict
        self.raw_title_dict = raw_title_dict
        self.stopwords = stopwords if stopwords is not None else set()
        self.processed_docs = {}
        self.processed_titles = {}
        if self.raw_text_dict:
            for doc_id, text in self.raw_text_dict.items():
                tokens = [token for token in self.document_preprocessor.tokenize(text) if token not in self.stopwords]
                title_tokens = [token for token in self.document_preprocessor.tokenize(self.raw_title_dict.get(doc_id, "")) if token not in self.stopwords]
                self.processed_docs[doc_id] = Counter(tokens)
                self.processed_titles[doc_id] = Counter(title_tokens)
    

    def build_vector_representation(self):
        vectors = {}
        for doc_id, doc_word_counts in self.processed_docs.items():
            query_parts = list(doc_word_counts.keys())
            features = self.feature_extractor.generate_features(doc_id, self.processed_docs, self.processed_titles, query_parts, self.raw_text_dict)
            vectors[doc_id] = features
        return vectors

    

    
class VectorFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 doc_category_info: dict[int, list[str]],
                 document_preprocessor: RegexTokenizer, stopwords: set[str],
                 recognized_categories: set[str],
                 ce_scorer: CrossEncoderScorer) -> None:
            
        self.document_index = document_index
        self.title_index = title_index
        self.doc_category_info = doc_category_info
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.recognized_categories= list(recognized_categories)
        self.ce_scorer = ce_scorer
        self.stats=self.document_index.get_statistics()
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.recognized_categories= set(self.stats["all_genres"])

    def get_roberta_features(self, text: str) -> torch.Tensor:
        inputs = self.roberta_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.roberta_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def get_article_length(self, docid: int) -> int:
        try:
            return self.document_index.statistics['docmap'][docid]['total_tokens']
        except KeyError:

            return 1 

    def get_title_length(self, docid: int) -> int:
        try:
            return self.title_index.statistics['docmap'][docid]['total_tokens']
        except KeyError:
            return 1  

    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        print("QUERY_PARTS", )
        total_tf = 0
        for term in query_parts:
            total_tf += math.log(word_counts.get(term, 0) + 1)
        print("TOTAL_TF", total_tf / (len(query_parts) if len(query_parts) > 0 else 1))
        return total_tf / (len(query_parts) if len(query_parts) > 0 else 1)



    def get_tf_idf(self, index: InvertedIndex, docid: int, word_counts: Dict[str, int], query_parts: List[str]) -> float:
        D = self.stats["number_of_documents"]
        tf_idf_score = 0
        for term in query_parts:
            tf = math.log(word_counts.get(term, 0) + 1)
            postings = index.get_postings(term)
            df_t_D = len(postings)
            idf = math.log((D / (df_t_D if df_t_D > 0 else 1))) + 1
            tf_idf_score += (tf * idf)
        return tf_idf_score / (len(query_parts) if len(query_parts) > 0 else 1)


    def get_BM25_score(self, docid: int, doc_word_counts: Dict[str, int], query_parts: List[str]) -> float:

        k1_value = 1.2
        k3_value = 8
        b_value = 0.75
        avgdl =  self.stats["mean_document_length"]
        bm25_score = 0
        N =  self.stats["number_of_documents"]
        for term in query_parts:
            f = doc_word_counts.get(term, 0)
            cq = query_parts.get(term, 0)
            postings = self.document_index.get_postings(term)
            df_t = len(postings)
            idf = math.log((N - df_t + 0.5) / (df_t + 0.5))
            if all(isinstance(val, defaultdict) for val in doc_word_counts.values()):
                doc_length = sum(sum(inner_dict.values()) for inner_dict in doc_word_counts.values())
            else:
                doc_length = sum(doc_word_counts.values())
            
            term_score = idf * (f * (k1_value + 1)) / (f + k1_value * (1 - b_value + b_value * doc_length / (avgdl if avgdl > 0 else 1)))
            query_component = (k3_value + 1) * cq / (k3_value + cq)
            
            bm25_score += term_score * query_component
        return bm25_score

        
    def get_pivoted_normalization_score(self, docid: int, doc_word_counts: Dict[str, int],
                                        query_parts: List[str]) -> float:
        
        try:
            N_value =  self.stats["number_of_documents"]
        except KeyError:
            N_value = 0

        try:
            D_value = self.document_index.get_doc_metadata(docid)["total_tokens"]
        except KeyError:
            D_value = 0

        try:
            avdl_value =  self.stats["mean_document_length"]
        except KeyError:
            avdl_value = 0

        b_value = 0.2  
        score_result = 0.0

        for term in query_parts:
            cd_wi = doc_word_counts.get(term, 0)
            df_wi = len(self.document_index.get_postings(term))
            cq_wi = query_parts.get(term, 0)
            f_term = doc_word_counts.get(term, 0)
            if f_term == 0:
                f_term = 0
            else:
                f_term = 1 + math.log(1 + math.log(f_term))
            tf_doc_denom = (1 - b_value + b_value * (D_value / (avdl_value if avdl_value > 0 else 1)))
            tf_doc_num = f_term
            tf_doc = tf_doc_num / (tf_doc_denom if tf_doc_denom > 0 else 1)
            if tf_doc_denom==0:
                tf_doc = tf_doc_num / (tf_doc_denom+1)
            else:
                tf_doc = tf_doc_num / tf_doc_denom
            if df_wi==0:
                idf_term = math.log((N_value + 1) / (df_wi+1))
            else:
                idf_term = math.log((N_value + 1) / df_wi)
        
            score_result += cq_wi * tf_doc * idf_term

        return score_result


    def get_document_categories(self, docid: int) -> list:

        categories = self.doc_category_info.get(docid, [])
        return [1 if category in categories else 0 for category in self.recognized_categories]


    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str], raw_text_dict=None) -> list:

        feature_vector = []
        feature_vector.append(self.get_article_length(docid))

        feature_vector.append(self.get_title_length(docid))

        feature_vector.append(len(query_parts))

        try:
            total_tf = self.get_tf(self.document_index, docid, Counter(doc_word_counts.get(docid, {})), Counter(query_parts))
            feature_vector.append(total_tf)
        except Exception as e:
            print("Error:", e)
            print("QUERY_PARTS", Counter(query_parts),"DOC_WORD_COUNTS",Counter(doc_word_counts.get(docid, {})))
            feature_vector.append(0)
            print("GOT ALL tf for docs")

        try:
            title_tf = self.get_tf(self.document_index, docid, Counter(title_word_counts.get(docid, {})), Counter(query_parts))
            feature_vector.append(title_tf)
        except:
            print("QUERY_PARTS", Counter(query_parts),"DOC_WORD_COUNTS",Counter(doc_word_counts.get(docid, {})))
            feature_vector.append(0)
            print("GOT ALL tf for titles")

        try:
            feature_vector.append(self.get_tf_idf(self.document_index,docid, Counter(doc_word_counts.get(docid, {})), Counter(query_parts)))
        except:
            print("QUERY_PARTS", Counter(query_parts),"DOC_WORD_COUNTS",Counter(doc_word_counts.get(docid, {})))
            feature_vector.append(0)
            print("GOT ALL tf_idf for docs")
        try:
            feature_vector.append(self.get_tf_idf(self.title_index,docid, Counter(title_word_counts.get(docid, {})), Counter(query_parts)))
        except:
            print("QUERY_PARTS", Counter(query_parts),"DOC_WORD_COUNTS",Counter(doc_word_counts.get(docid, {})))
            feature_vector.append(0)
            print("GOT ALL tf_idf for titles") 
        try:
            feature_vector.append(self.get_BM25_score(docid, Counter(doc_word_counts.get(docid, {})), Counter(query_parts)))
        except:
            print( "QUERY_PARTS", Counter(query_parts),"DOC_WORD_COUNTS",Counter(doc_word_counts.get(docid, {})))
            feature_vector.append(0)
            print("GOT ALL bm25 for docs")
                              
        try:
            feature_vector.append(self.get_pivoted_normalization_score(docid, Counter(doc_word_counts.get(docid, {})), Counter(query_parts)))
        except:
            print("QUERY_PARTS", Counter(query_parts),"DOC_WORD_COUNTS",Counter(doc_word_counts.get(docid, {})))
            feature_vector.append(0)
            print("GOT ALL pivoted for docs")


        cat_vec= self.get_document_categories(docid)
        feature_vector.extend(cat_vec)

        text = raw_text_dict.get(docid, "")  
        roberta_features = self.get_roberta_features(text)
        feature_vector.extend([feature for feature in roberta_features.tolist()])

        return feature_vector


        


