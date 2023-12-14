from SearchEngine import SearchEngine   
from Recommender import Recommender
from dotenv import load_dotenv
import os
load_dotenv()



class app:
    def __init__(self) -> None:
        
        self.searchengine=SearchEngine()
        self.recommender=Recommender("Processed_Data", "Vectorized_books")

    def query(self,query="search",user_book_ids=None):
        print("I a in query")
        if query:
            if user_book_ids:
                top_100_book_ids=self.searchengine.query_with_l2r_with_BM25(query,True,user_book_ids)
                recommendations=self.recommender.get_recommendations(user_book_ids, top_100_book_ids)
            else:
                print("IAM HERE")
                top_100_book_ids=self.searchengine.query_with_l2r_with_BM25(query)
app=app()
app.query("search")
