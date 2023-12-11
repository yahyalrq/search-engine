from sentence_transformers import SentenceTransformer, util
from numpy import ndarray
from ranker import Ranker
import numpy as np
import torch

class VectorRanker(Ranker):
    """  
        Initializes a VectorRanker object.

        Args:
            bi_encoder_model_name: The name of a huggingface model to use for initializing a 'SentenceTransformer'
            encoded_docs: A matrix where each row is an already-encoded document, encoded using the same encoded
                as specified with bi_encoded_model_name
            row_to_docid: A list that is a mapping from the row number to the document id that row corresponds to
                the embedding

        Using zip(encoded_docs, row_to_docid) should give you the mapping between the docid and the embedding.
        """
    def __init__(self, bi_encoder_model_name: str, encoded_docs: ndarray,
                 row_to_docid: list[int]) -> None:
        self.bi_encoder = SentenceTransformer(bi_encoder_model_name, device='cpu')
        self.encoded_docs = encoded_docs
        self.docid_to_row = {doc_id: idx for idx, doc_id in enumerate(row_to_docid)}
    


    def query(self, query: str, pseudofeedback_num_docs=0,
              pseudofeedback_alpha=0.8, pseudofeedback_beta=0.2, user_id=None) -> list[tuple[int, float]]:
        """
        Encodes the query and then scores the relevance of the query with all the documents.

        Args:
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A sorted list of tuples containing the document id and its relevance to the query,
            with most relevant documents first
        """
        if not query.strip():
            return []
        
        query_vec = self.bi_encoder.encode(query, convert_to_numpy=True)

        if pseudofeedback_num_docs > 0:
            print("IN PSEUDO FEEDBACCCCK")
            initial_similarities = np.dot(self.encoded_docs, query_vec)

            top_docs_indices = initial_similarities.argsort()[-pseudofeedback_num_docs:][::-1]

            top_doc_vectors = self.encoded_docs[top_docs_indices]
            avg_vector = np.mean(top_doc_vectors, axis=0)

            query_vec = pseudofeedback_alpha * query_vec + pseudofeedback_beta * avg_vector

        similarities = np.dot(self.encoded_docs, query_vec)
        
        doc_scores = [(doc_id, similarities[self.docid_to_row[doc_id]]) for doc_id in self.docid_to_row]

        ranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        return ranked_docs
