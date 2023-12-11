from models import BaseSearchEngine, SearchResponse

# your library imports go here
from document_preprocessor import RegexTokenizer
from indexing import Indexer, IndexType
from ranker import Ranker, BM25
from l2r import L2RRanker

class SearchEngine(BaseSearchEngine):
    def __init__(self) -> None:
        pass
      
    def search(self, query: str) -> list[SearchResponse]:
        results = self.ranker.query(query)
        return [SearchResponse(id=idx+1, docid=result[0], score=result[1]) for idx, result in enumerate(results)]


def initialize():
    search_obj = SearchEngine()
    return search_obj
