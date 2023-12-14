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
import json
import pickle
import os
import torch
from transformers import RobertaModel, RobertaTokenizer

class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: RegexTokenizer, stopwords: set[str], ranker,
                 feature_extractor: 'L2RFeatureExtractor',raw_text_dict=None, raw_title_dict=None ) -> None:
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.scorer = ranker
        self.feature_extractor = feature_extractor
        self.model = LambdaMART()
        self.raw_text_dict=raw_text_dict
        self.raw_title_dict=raw_title_dict
        self.raw_text_dict=raw_text_dict
        self.raw_title_dict=raw_title_dict
        self.processed_docs = {}
        self.processed_titles = {}
        if self.raw_text_dict:
            for doc_id in self.raw_text_dict:
                tokens = [token for token in self.document_preprocessor.tokenize(self.raw_text_dict[doc_id]) if token not in self.stopwords]
                title_tokens = [token for token in self.document_preprocessor.tokenize(self.raw_title_dict[doc_id]) if token not in self.stopwords]
                self.processed_docs[doc_id] = Counter(tokens)
                self.processed_titles[doc_id] = Counter(title_tokens)
        
                   
    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]):
        print("IN PREPARE TRAINING DATA")
        X = []
        y = []
        qgroups = []
        for query, relevant_pairs in query_to_document_relevance_scores.items():
            query_tokens = self.document_preprocessor.tokenize(query)
            query_parts=Counter(query_tokens)
            if self.raw_text_dict:
                doc_word_counts = {}
                title_word_counts = {}
                for term in query_parts:
                    postings = self.document_index.get_postings(term)
                    for doc_id, _ in postings:
                        if doc_id not in doc_word_counts:
                            doc_word_counts[doc_id] = self.processed_docs[doc_id]
                            title_word_counts[doc_id] = self.processed_titles[doc_id]

                for docid, relevance in relevant_pairs:
                    features = self.feature_extractor.generate_features(docid, doc_word_counts, title_word_counts,query_tokens, query, self.raw_text_dict)
                    X.append(features)
                    y.append(relevance)
            else:
                doc_term_counts_content = L2RRanker.accumulate_doc_term_counts(self.document_index, query_parts)
                doc_term_counts_title = L2RRanker.accumulate_doc_term_counts(self.title_index, query_parts)

                for docid, relevance in relevant_pairs:
                    features = self.feature_extractor.generate_features(docid, doc_term_counts_content, doc_term_counts_title,query_parts, query)
                    X.append(features)
                    y.append(relevance)

            qgroups.append(len(relevant_pairs))
        return X, y, qgroups

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> Dict[int, Dict[str, int]]:

        counts_term_doc = defaultdict(Counter)
        for token in query_parts: 
            postings = index.get_postings(token)
            for doc_id, freq in postings:
                counts_term_doc[doc_id][token] += freq

        counts_term_doc = {doc_id: dict(terms) for doc_id, terms in counts_term_doc.items()}
        return counts_term_doc
    
    def train(self, training_data_filename: str) -> None:
        current_path = os.getcwd()
        model_filename = os.path.join(current_path, 'full_model_ourrel.pkl') 

        if os.path.exists(model_filename):
            print("Loading model from file")
            with open(model_filename, 'rb') as file:
                self.model = pickle.load(file)
                print("MODEL LOAAAAAAAADEEEEED")
        else:
            df = pd.read_csv(training_data_filename)
            relevance_scores_of_query_to_document = df.groupby('query').apply(lambda group: list(zip(group['docid'], group['rel']))).to_dict()
            
            print("Start Prep training data")
            print("Number of relevance_scores_of_query_to_document", len(relevance_scores_of_query_to_document))
            X, y, qgroups = self.prepare_training_data(relevance_scores_of_query_to_document)
            
            print("End Prep training data")
            print("Model started fitting")
            self.model.fit(X_train=X, y_train=y, qgroups_train=qgroups)
            print("Model finished fitting")

            with open(model_filename, 'wb') as file:
                pickle.dump(self.model, file)
                print(f"Model saved to {model_filename}")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)
    

    
    def _modify_query_with_pseudofeedback(self, query_tokens, initial_scores, pseudofeedback_num_docs, doc_word_counts, pseudofeedback_alpha, pseudofeedback_beta):
        print("START TO MODIFY QUERY")
        # Sort and select top documents
        top_docs = sorted(initial_scores, key=initial_scores.get, reverse=True)[:pseudofeedback_num_docs]

        # Count initial query tokens
        query_token_counts = Counter(query_tokens)

        # Initialize term weights
        term_weights = defaultdict(float)
        reciprocal = 1 / pseudofeedback_num_docs  # Pre-calculate reciprocal

        # Calculate term weights from top documents
        for doc_id in top_docs:
            for term, count in doc_word_counts[doc_id].items():
                term_weights[term] += count * reciprocal

        # Modify query with pseudofeedback
        modified_query = Counter()
        for term in term_weights:
            original_weight = pseudofeedback_alpha * query_token_counts[term]
            feedback_weight = pseudofeedback_beta * term_weights[term]
            modified_query[term] = original_weight + feedback_weight
        print("FINISHED MODIFYING QUERY")
        return modified_query


    def query(self, query: str, pseudo_feedback_num_docs=0) -> list[tuple[int, float]]:
        if not query: 
            return None
        if not query.strip():
            return 
        query_tokens = self.document_preprocessor.tokenize(query)
        query_tokens = [q for q in query_tokens if q not in self.stopwords]
    
        query_parts=Counter(query_tokens)
        if self.raw_text_dict:
            query_parts=Counter(query_tokens)
            doc_word_counts={}
            title_word_counts = {}
            document_scores={}
            for term in query_parts:
                postings = self.document_index.get_postings(term)
                for doc_id, _ in postings:
                    if doc_id not in doc_word_counts:
                        doc_word_counts[doc_id] = self.processed_docs[doc_id]
                        title_word_counts[doc_id] = self.processed_titles[doc_id]
                        score=self.feature_extractor.get_BM25_score(doc_id, doc_word_counts.get(doc_id, {}), query_parts)
                        #score=self.scorer.scorer.score(doc_id, doc_word_counts.get(doc_id, {}), query_parts)
                        document_scores[doc_id] = score

                        
            if pseudo_feedback_num_docs > 0:
                query_parts = self._modify_query_with_pseudofeedback(query_tokens, document_scores, pseudo_feedback_num_docs, doc_word_counts, pseudofeedback_alpha=0.8, pseudofeedback_beta=0.2)
                for term in query_parts:
                    postings = self.document_index.get_postings(term)
                    for doc_id, _ in postings:
                        if doc_id not in doc_word_counts:
                            doc_word_counts[doc_id] = self.processed_docs[doc_id]
                            title_word_counts[doc_id] = self.processed_titles[doc_id]
                            #score=self.feature_extractor.get_BM25_score(doc_id, doc_word_counts.get(doc_id, {}), query_parts)
                            score=self.scorer.scorer.score(doc_id, doc_word_counts.get(doc_id, {}), query_parts)
                            document_scores[doc_id] = score
            
            top_document_ids = sorted(document_scores, key=document_scores.get, reverse=True)[:100]
            print("start to generate features")
            X = []
            for doc_id in top_document_ids:
                features = self.feature_extractor.generate_features(doc_id, doc_word_counts, title_word_counts, query_tokens, query,self.raw_text_dict)
                X.append(features)
        else:
            
            doc_term_counts_content = self.accumulate_doc_term_counts(self.document_index, query_tokens)
            doc_term_counts_title = self.accumulate_doc_term_counts(self.title_index, query_tokens)
            document_scores = {}
            for doc_id, term_counts in doc_term_counts_content.items():
                document_scores[doc_id] = self.scorer.scorer.score(doc_id, term_counts, Counter(query_tokens))
            top_document_ids = sorted(document_scores.keys(), key=lambda x: document_scores[x], reverse=True)[:100]

            X = []
            for doc_id in top_document_ids:
                features = self.feature_extractor.generate_features(doc_id, doc_term_counts_content, doc_term_counts_title, query_tokens, query)
                X.append(features)


        predictions = self.predict(X)
        ranked_doc_ids = [doc for _, doc in sorted(zip(predictions, top_document_ids), reverse=True)]

        other_doc_ids = [doc for doc in document_scores.keys() if doc not in top_document_ids]
        ranked_doc_ids.extend(sorted(other_doc_ids, key=lambda x: document_scores[x], reverse=True))

        ranked_docs = [{'doc_id': doc_id, 'score': document_scores[doc_id]} for doc_id in ranked_doc_ids if doc_id in document_scores]

        return ranked_docs
        
    def vector_query(self, query: str, docid_to_index, pseudo_feedback_num_docs=0) -> list[tuple[int, float]]:
        if not query: 
            return None
        if not query.strip():
            return 
        query_tokens = self.document_preprocessor.tokenize(query)
        query_tokens = [q for q in query_tokens if q not in self.stopwords]
        print("Tokenized the query")

        document_scores={}
        print("VECTOR SCORING", query)
        if pseudo_feedback_num_docs>0:
            res_list = self.scorer.query(query,pseudo_feedback_num_docs)
        else:
            res_list = self.scorer.query(query)
        for doc_id, score in res_list:
            document_scores[int(doc_id)] = score
        doc_ids, similarity_scores = zip(*res_list)
        doc_indices = [docid_to_index[doc_id] for doc_id in doc_ids]
        sorted_indices = np.argsort(similarity_scores)[::-1]
        sorted_doc_indices = np.array([doc_indices[i] for i in sorted_indices])
        top_document_ids = sorted(sorted_doc_indices, key=lambda x: sorted_doc_indices[x], reverse=True)[:100]
        if self.raw_text_dict:
            query_parts=Counter(query_tokens)
            doc_word_counts={}
            title_word_counts = {}

            for term in query_parts:
                postings = self.document_index.get_postings(term)
                for doc_id, _ in postings:
                    if doc_id not in doc_word_counts:
                        doc_word_counts[doc_id] = self.processed_docs[doc_id]
                        title_word_counts[doc_id] = self.processed_titles[doc_id]
            X = []
            for doc_id in top_document_ids:
                features = self.feature_extractor.generate_features(doc_id,doc_word_counts, title_word_counts, query_tokens, query)
                X.append(features)
        else:
            
            doc_term_counts_content = self.accumulate_doc_term_counts(self.document_index, query_tokens)
            doc_term_counts_title = self.accumulate_doc_term_counts(self.title_index, query_tokens)
            X = []
            for doc_id in top_document_ids:
                features = self.feature_extractor.generate_features(doc_id, doc_term_counts_content, doc_term_counts_title, query_tokens, query)
                X.append(features)

        predictions = self.predict(X)
        ranked_doc_ids = [doc for _, doc in sorted(zip(predictions, top_document_ids), reverse=True)]

        other_doc_ids = [doc for doc in document_scores.keys() if doc not in top_document_ids]
        ranked_doc_ids.extend(sorted(other_doc_ids, key=lambda x: document_scores[x], reverse=True))

        ranked_docs = [{'doc_id': doc_id, 'score': document_scores[doc_id]} for doc_id in ranked_doc_ids if doc_id in document_scores]
        return ranked_docs

    
class L2RFeatureExtractor:
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
        self.ce_scorer = ce_scorer
        self.stats=self.document_index.get_statistics()
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.recognized_categories= set(self.stats["all_genres"])


    def get_combined_roberta_features(self, query: str, doc_text: str) -> torch.Tensor:

        combined_text = query + " " + doc_text 
        inputs = self.roberta_tokenizer(combined_text, return_tensors="pt", max_length=512, truncation=True)
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
        total_tf = 0
        for term in query_parts:
            total_tf += math.log(word_counts.get(term, 0) + 1)
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
        score= [1 if category in categories else 0 for category in self.recognized_categories]
        return score


    def get_cross_encoder_score(self, docid: int, query: str) -> float:

        return self.ce_scorer.score(docid, query)


    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str],
                          query: str,  raw_text_dict=None) -> list:

        feature_vector = []
        feature_vector.append(self.get_article_length(docid))

        feature_vector.append(self.get_title_length(docid))

        feature_vector.append(len(query_parts))


        try:
            total_tf = self.get_tf(self.document_index, docid, Counter(doc_word_counts.get(docid, {})), Counter(query_parts))
            feature_vector.append(total_tf)
        except:
            print("QUERY", query, "QUERY_PARTS", Counter(query_parts),"DOC_WORD_COUNTS",Counter(doc_word_counts.get(docid, {})))
            feature_vector.append(0)
            print("GOT ALL tf for docs")

        try:
            title_tf = self.get_tf(self.document_index, docid, Counter(title_word_counts.get(docid, {})), Counter(query_parts))
            feature_vector.append(title_tf)
        except:
            print("QUERY", query, "QUERY_PARTS", Counter(query_parts),"DOC_WORD_COUNTS",Counter(doc_word_counts.get(docid, {})))
            feature_vector.append(0)
            print("GOT ALL tf for titles")

        try:
            feature_vector.append(self.get_tf_idf(self.document_index,docid, Counter(doc_word_counts.get(docid, {})), Counter(query_parts)))
        except:
            print("QUERY", query, "QUERY_PARTS", Counter(query_parts),"DOC_WORD_COUNTS",Counter(doc_word_counts.get(docid, {})))
            feature_vector.append(0)
            print("GOT ALL tf_idf for docs")
        try:
            feature_vector.append(self.get_tf_idf(self.title_index,docid, Counter(title_word_counts.get(docid, {})), Counter(query_parts)))
        except:
            print("QUERY", query, "QUERY_PARTS", Counter(query_parts),"DOC_WORD_COUNTS",Counter(doc_word_counts.get(docid, {})))
            feature_vector.append(0)
            print("GOT ALL tf_idf for titles") 
        # TODO: BM25
        try:
            feature_vector.append(self.get_BM25_score(docid, Counter(doc_word_counts.get(docid, {})), Counter(query_parts)))
        except:
            print("QUERY", query, "QUERY_PARTS", Counter(query_parts),"DOC_WORD_COUNTS",Counter(doc_word_counts.get(docid, {})))
            feature_vector.append(0)
            print("GOT ALL bm25 for docs")
                              
        try:
            feature_vector.append(self.get_pivoted_normalization_score(docid, Counter(doc_word_counts.get(docid, {})), Counter(query_parts)))
        except:
            print("QUERY", query, "QUERY_PARTS", Counter(query_parts),"DOC_WORD_COUNTS",Counter(doc_word_counts.get(docid, {})))
            feature_vector.append(0)
            print("GOT ALL pivoted for docs")


        # Document Categories
        cat_vec= self.get_document_categories(docid)
        feature_vector.extend(cat_vec)
        text = raw_text_dict.get(docid, "")
        combined_roberta_features = self.get_combined_roberta_features(query, text)
        combined_roberta_features = combined_roberta_features.view(-1)
        feature_vector.extend([feature for feature in combined_roberta_features.tolist()])
        return feature_vector


class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 10,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.04,
            'max_depth': -1,
            # NOTE: You might consider setting this parameter to a higher value equal to 
            # the number of CPUs on your machine for faster training
            "n_jobs": 6, 
        }

        if params:
            default_params.update(params)
        self.model = lightgbm.LGBMRanker(**default_params)

        # TODO: initialize the LGBMRanker with the provided parameters and assign as a field of this class        

    def fit(self,  X_train, y_train, qgroups_train):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples.
            y_train (array-like): Target values.
            qgroups_train (array-like): Query group sizes for training data.

        Returns:
            self: Returns the instance itself.
        """
        
        # TODO: fit the LGBMRanker's parameters using the provided features and labels
        self.model.fit(X_train, y_train, group=qgroups_train)
        return self

    def predict(self, featurized_docs):
        """
        Predicts the relevance (ranks) of the featurized documents and return the values.

        Args:
            featurized_docs (array-like): 
                A list of featurized documents where each document is a list of its features
                All documents should have the same length.

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """

        # TODO: Generating the predicted values using the LGBMRanker
        return self.model.predict(featurized_docs)
        pass



""" 
doc_word_counts={}
title_word_counts={}
document_scores = {}
for term in query_parts:
    postings = self.document_index.get_postings(term)
    for posting in postings:
        doc_id=posting[0]
        tokens = [token if token not in self.stopwords else None for token in self.document_preprocessor.tokenize(self.raw_text_dict[doc_id])]
        titletokens = [token if token not in self.stopwords else None for token in self.document_preprocessor.tokenize(self.raw_title_dict[doc_id])] 
        doc_word_counts[doc_id] = Counter(tokens)
        title_word_counts[doc_id]= Counter(titletokens)
        if pseudo_feedback_num_docs>0:
            score = self.scorer.scorer.score(doc_id, doc_word_counts[doc_id], query_parts, pseudo_feedback_num_docs)
        else:
            score = self.scorer.scorer.score(doc_id, doc_word_counts[doc_id], query_parts)
        document_scores[doc_id] = score"""