from nltk.tokenize import RegexpTokenizer
import torch
"""from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import json
import gzip
import seaborn as sns
import matplotlib.pyplot as plt"""
from collections import defaultdict
from tqdm import tqdm

from typing import Dict, List, Union, Set, Tuple


class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: List[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3.
        """
        # TODO: Save arguments that are needed as fields of this class

    def find_and_replace_mwes(self, input_tokens: List[str]) -> List[str]:
        """
        IGNORE THIS PART; NO NEED TO IMPLEMENT THIS SINCE NO MULTI-WORD EXPRESSION PROCESSING IS TO BE USED.
        For the given sequence of tokens, finds any recognized multi-word expressions in the sequence
        and replaces that subsequence with a single token containing the multi-word expression.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens containing processed multi-word expressions
        """
        # NOTE: You shouldn't implement this in homework 
        raise NotImplemented("MWE is not supported")
    
    def postprocess(self, input_tokens: List[str]) -> List[str]:
        """
        Performs any set of optional operations to modify the tokenized list of words such as
        lower-casing and returns the modified list of tokens.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens processed by lower-casing depending on the given condition
        """
        re_tokens=[]
        for token in input_tokens:
            token=token.lower()
            re_tokens.append(token)
        return re_tokens

    
    def tokenize(self, text: str) -> List[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        raise NotImplementedError('tokenize() is not implemented in the base class; please use a subclass')


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex="\w+", lowercase: bool = True, multiword_expressions: List[str] = None) -> None:
        """
        Uses NLTK's RegexpTokenizer to tokenize a given string.

        Args:
            token_regex: Use the following default regular expression pattern: '\\w+'
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        # TODO: Save a new argument that is needed as a field of this class
        # TODO: Initialize the NLTK's RegexpTokenizer 

    def tokenize(self, text: str) -> List[str]:
        self.tokenizer = RegexpTokenizer('\\w+')
        tokens = self.tokenizer.tokenize(text)
        tokens=self.postprocess(tokens)
        return tokens

# TODO (HW3): Take in a doc2query model and generate queries from a piece of text
# Note: This is just to check you can use the models;
#       for downstream tasks such as index augmentation with the queries, use doc2query.csv
class Doc2QueryAugmenter:
    """
    This class is responsible for generating queries for a document.
    These queries can augment the document before indexing.

    MUST READ: https://huggingface.co/doc2query/msmarco-t5-base-v1

    OPTIONAL reading
        1. Document Expansion by Query Prediction (Nogueira et al.): https://arxiv.org/pdf/1904.08375.pdf
    """
    def __init__(self, doc2query_model_name: str = 'doc2query/msmarco-t5-base-v1') -> None:
        """
        Creates the T5 model object and the corresponding dense tokenizer.
        
        Args:
            doc2query_model_name: The name of the T5 model architecture used for generating queries

        """
        self.device = torch.device('cpu')  # Do not change this unless you know what you are doing
        self.tokenizer = T5Tokenizer.from_pretrained(doc2query_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(doc2query_model_name).to(self.device)

        # TODO: Create the dense tokenizer and query generation model using HuggingFace transformers

    def get_queries(self, document: str, n_queries: int = 5, prefix_prompt: str = '') -> List[str]:
        """
        Steps
            1. Use the dense tokenizer/encoder to create the dense document vector.
            2. Use the T5 model to generate the dense query vectors (you should have a list of vectors).
            3. Decode the query vector using the tokenizer/decode to get the appropriate queries.
            4. Return the queries.
         
            Ensure you take care of edge cases.
         
        OPTIONAL (DO NOT DO THIS before you finish the assignment):
            Neural models are best performing when batched to the GPU.
            Try writing a separate function which can deal with batches of documents.
        
        Args:
            document: The text from which queries are to be generated
            n_queries: The total number of queries to be generated
            prefix_prompt: An optional parameter
                Some models are not fine-tuned to generate queries.
                So we need to add a prompt to coax the model into generating queries.
                This string enables us to create a prefixed prompt to generate queries for the models.
                Prompt-engineering: https://en.wikipedia.org/wiki/Prompt_engineering
        
        Returns:
            A list of query strings generated from the text
        """
        document_max_token_length = 400 
        top_p = 0.85
        input_encoded_tokens = self.tokenizer.encode(prefix_prompt + document, return_tensors='pt', max_length=document_max_token_length, truncation=True).to(self.device)
        output = self.model.generate(input_ids=input_encoded_tokens, max_length=64, num_return_sequences=n_queries, top_p=top_p)
        
        queries = [self.tokenizer.decode(generated_output, skip_special_tokens=True) for generated_output in output]
        return queries


