

import numpy as np
from collections import Counter, defaultdict
from sentence_transformers import CrossEncoder
from indexing import InvertedIndex
from typing import Dict, List, Union, Set
import math
from document_preprocessor import RegexTokenizer
import re
from nltk.tokenize import RegexpTokenizer
    
class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    # TODO: implement this class properly. This is responsible for returning a list of sorted relevant documents.
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str], 
                 scorer, raw_text_dict: dict[int,str]) -> None:
        """
        Initializes the state of the Ranker object.

        NOTE: Previous homeworks had you passing the class of the scorer to this function.
            This has been changed as it created a lot of confusion.
            You should now pass an instantiated RelevanceScorer to this function.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict
        parameters = {'b': 0.75, 'k1': 1.2, 'k3': 8}
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
        self.processed_docs = {}
        if self.raw_text_dict:
            for doc_id in self.raw_text_dict:
                tokens = [token for token in self.tokenize(self.raw_text_dict[doc_id]) if token not in self.stopwords]
                self.processed_docs[doc_id] = Counter(tokens)
    

    def query(self, query: str, pseudofeedback_num_docs=0, pseudofeedback_alpha=0.8,
              pseudofeedback_beta=0.2, user_id=None) -> list[tuple[int, float]]:
        query_tokens = [token for token in self.tokenize(query) if token not in self.stopwords]
        query_parts=Counter(query_tokens)
      
        self.doc_word_counts = {}
        document_scores = defaultdict(float)

        for term in query_parts:
            postings = self.index.get_postings(term)
            for doc_id, _ in postings:
                if doc_id not in self.doc_word_counts:
                    self.doc_word_counts[doc_id] = self.processed_docs[doc_id]
                    score = self.scorer.score(doc_id, self.doc_word_counts[doc_id], query_parts)
                    document_scores[doc_id] = score


        if pseudofeedback_num_docs > 0:
            query_parts = self._modify_query_with_pseudofeedback(query_tokens,document_scores, 
                                                                    pseudofeedback_num_docs, 
                                                                    pseudofeedback_alpha, 
                                                                    pseudofeedback_beta)

            for term in query_parts:
                postings = self.index.get_postings(term)
                for posting in postings:
                    doc_id=posting[0]
                    tokens = [token if token not in self.stopwords else None for token in self.tokenize(self.raw_text_dict[doc_id])]
                    self.doc_word_counts[doc_id] = Counter(tokens)
                    score = self.scorer.score(doc_id, self.doc_word_counts[doc_id], query_parts)
                    document_scores[doc_id] = score
        document_scores=sorted(document_scores.items(), key= lambda x:x[1], reverse=True)
        ranked_docs = [{'doc_id': doc_id, 'score': score} for doc_id, score in document_scores]
        return ranked_docs

class PersonalizedBM25(Ranker):
    def __init__(self, index, relevant_doc_index, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}):
        self.index = index
        self.relevant_doc_index = relevant_doc_index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
        self.stats=self.index.get_statistics()
        self.rel_stats=self.relevant_doc_index.get_statistics()
        self.R = self.rel_stats["number_of_documents"] 
        

    def score(self, docid, doc_word_counts, query_word_counts):
        document_mean_length = self.stats["mean_document_length"]
        num_documents = self.stats["number_of_documents"]
        personalized_score = 0

        for query_term, query_term_count in query_word_counts.items():
            term_frequency = doc_word_counts.get(query_term, 0)
            term_postings = self.index.get_postings(query_term)
            doc_frequency_term = len(term_postings)
            relevant_postings = self.relevant_doc_index.get_postings(query_term)
            ri = len(relevant_postings) 

            inverse_doc_freq = math.log(((ri + 0.5) * (num_documents - doc_frequency_term - self.R + ri + 0.5)) /
                                        ((doc_frequency_term - ri + 0.5) * (self.R - ri + 0.5)))


            if all(isinstance(value, defaultdict) for value in doc_word_counts.values()):
                total_doc_length = sum(sum(inner_dict.values()) for inner_dict in doc_word_counts.values())
            else:
                total_doc_length = sum(doc_word_counts.values())
            individual_term_score = inverse_doc_freq * (term_frequency * (self.k1 + 1)) / (term_frequency + self.k1 * (1 - self.b + self.b * total_doc_length / document_mean_length))
            query_term_weight = (self.k3 + 1) * query_term_count / (self.k3 + query_term_count)
            
            personalized_score += individual_term_score * query_term_weight
        return personalized_score


    
class BM25(Ranker):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
        self.stats=self.index.get_statistics()

    def score(self, docid: int, doc_word_counts: Dict[str, int], query_parts: List[str]) -> float:
        document_mean_length = self.stats["mean_document_length"]  
        num_documents = self.stats["number_of_documents"]
        calculated_bm25_score = 0
        query_parts=Counter(query_parts)

        for query_term in query_parts:
            term_frequency = doc_word_counts.get(query_term, 0)
            query_term_count = query_parts.get(query_term, 0)
            term_postings = self.index.get_postings(query_term)
            doc_frequency_term = len(term_postings)
            inverse_doc_freq = math.log((num_documents - doc_frequency_term + 0.5) / (doc_frequency_term + 0.5))
            
            if all(isinstance(value, defaultdict) for value in doc_word_counts.values()):
                total_doc_length = sum(sum(inner_dict.values()) for inner_dict in doc_word_counts.values())
            else:
                total_doc_length = sum(doc_word_counts.values())
            
            individual_term_score = inverse_doc_freq * (term_frequency * (self.k1 + 1)) / (term_frequency + self.k1 * (1 - self.b + self.b * total_doc_length / document_mean_length))
            query_term_weight = (self.k3 + 1) * query_term_count / (self.k3 + query_term_count)
            
            calculated_bm25_score += individual_term_score * query_term_weight

        return calculated_bm25_score



    




class PivotedNormalization(Ranker):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: Dict[str, int], query_parts: List[str]):
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score
            # hint: 
            ## term_frq will always be >0
            ## doc_frq will always be >0 since only looking at terms that are in both query and doc

        # 4. Return the score
        num_documents = self.index.get_statistics()["number_of_documents"]
        document_token_count = self.index.get_doc_metadata(docid)["total_tokens"]
        mean_document_length = self.index.get_statistics()["mean_document_length"]
        bm25_b_value = self.b
        calculated_score = 0.0

        for query_term in query_parts:
            term_count_in_doc = doc_word_counts.get(query_term, 0)
            doc_frequency_for_term = len(self.index.get_postings(query_term))
            count_of_query_term = query_parts.get(query_term, 0)

            # Calculate term frequency (TF)
            term_frequency = doc_word_counts.get(query_term, 0)
            if term_frequency == 0:
                term_frequency_adjusted = 0
            else:
                term_frequency_adjusted = 1 + math.log(1 + math.log(term_frequency))
            tf_numerator = term_frequency_adjusted
            tf_denominator = (1 - bm25_b_value + bm25_b_value * (document_token_count / mean_document_length))
            term_frequency_normalized = tf_numerator / tf_denominator

            # Calculate inverse document frequency (IDF)
            inverse_doc_freq = math.log((num_documents + 1) / doc_frequency_for_term)

            # Update the score
            calculated_score += count_of_query_term * term_frequency_normalized * inverse_doc_freq

        return calculated_score



class TF_IDF(Ranker):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: Dict[str, int], query_parts: List[str]) -> float:
        document_count = self.index.get_statistics()["number_of_documents"] 
        calculated_tf_idf_score = 0

        for query_term in query_parts:
            term_frequency_log = math.log(doc_word_counts.get(query_term, 0) + 1)
            term_postings = self.index.get_postings(query_term)
            document_frequency_for_term = len(term_postings)
            inverse_document_frequency = math.log((document_count / document_frequency_for_term)) + 1 if document_frequency_for_term else 0
            calculated_tf_idf_score += (term_frequency_log * inverse_document_frequency)

        return calculated_tf_idf_score


class CrossEncoderScorer:
    def __init__(self, raw_text_dict: dict[int, str], cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.
        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the document content
            cross_encoder_model_name: The name of a cross-encoder model
        """
        self.raw_text_dict = raw_text_dict
        self.model = CrossEncoder(cross_encoder_model_name)

    def score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.
        
        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """

        if docid not in self.raw_text_dict or not query:
            return 0.0

        document_content = self.raw_text_dict[docid]
        score = self.model.predict([(query, document_content)])
        
        return score[0]




class RelevanceScorer:
    """
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    """
    # TODO: Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25,
    #  PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index: InvertedIndex, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        TODO (HW4): Note that the `query_word_counts` is now a dictionary of words and their counts.
            This is changed from the previous homeworks.
        """
        raise NotImplementedError

class WordCountCosineSimilarity(Ranker):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query

        # 2. Return the score
        return score  # (`score` should be defined in your code; you can call it whatever you want)


class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query_parts, compute score

        # 4. Return the score
        return score  # (`score` should be defined in your code; you can call it whatever you want)



"""
self.doc_word_counts = {} 
doc_scores = defaultdict(float)
query_parts=Counter(query_tokens)
for term in query_parts:
    postings = self.index.get_postings(term)
    for posting in postings:
        doc_id=posting[0]
        tokens = [token for token in self.tokenize(self.raw_text_dict[doc_id]) ]
        self.doc_word_counts[doc_id] = Counter(tokens)
        score = self.scorer.score(doc_id, self.doc_word_counts[doc_id], query_parts)
        doc_scores[doc_id] = score"""