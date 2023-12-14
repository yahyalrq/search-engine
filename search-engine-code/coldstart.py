import streamlit as st
import pymongo
import json
from bson import json_util
from streamlit.components.v1 import html
import sys
import os
from app import app
from bson import ObjectId
from streamlit_carousel import carousel

if 'selected_books_ids' not in st.session_state:
        st.session_state['selected_books_ids'] = []
if 'user_preference' not in st.session_state:
    st.session_state['user_preference'] = []
if "carousel_items" not in st.session_state:
    st.session_state['carousel_items'] = []

# Determine the absolute path of the parent directory
#parent_dir = os.path.dirname(os.path.abspath(__file__))
#print("HELLOOOOOOO: ", parent_dir)
#project_dir = os.path.dirname(parent_dir)

# Add the parent directory to sys.path
#sys.path.append(project_dir)

# Now you can import the app class from app.py
#from search_engine_code.app import app


mongo_conn_str = "mongodb+srv://bmazari:Rayane80@ir-final.8vivaaw.mongodb.net/?retryWrites=true&w=majority"

# Connect to MongoDB
client = pymongo.MongoClient(mongo_conn_str)
db = client['Processed_Data']
collection = db['processed_books']

@st.cache_resource
def create_app():
    return app()

# Access your database and collection
def retrieve_genres(collection):
    
    # Aggregation pipeline to count and find the most frequent genre
    pipeline = [
        {"$unwind": "$genres"},  # Unwind the genres array
        {"$group": {"_id": "$genres", "count": {"$sum": 1}}},  # Group by genre and count occurrences
        {"$sort": {"count": -1}},  # Sort genres in descending order of count
        {"$limit": 10}  # Optional: limit to the top genre only
    ]
    # Execute the aggregation pipeline
    frequent_genre = collection.aggregate(pipeline)
    genres= []
    for genre in frequent_genre:
        genres.append(genre['_id']) # The genre name
    return genres



def retrieve_books(preferences, collection):
    books_by_genre = {}

    # Iterate through each genre in the preferences
    for genre in preferences:
        # Query to find books with the current genre
        query = {"genres": genre}
        fields_to_retrieve = {"_id": 1, "book_id": 1, "title": 1, "author": 1, "language": 1, "publisher": 1, "year_published": 1, "description": 1}

        # Execute the query
        book_documents = collection.find(query, fields_to_retrieve).limit(10)

        # Add the results to the dictionary
        books_by_genre[genre] = [book for book in book_documents]
    return books_by_genre

#preferences = ['Fiction', 'Fantasy', 'Romance', 'Young Adult', 'Nonfiction', 'Mystery', 'Contemporary', 'Historical Fiction', 'Childrens', 'Classics']
#retrieve_books(preferences, collection)

def book_carousel_selection(books_by_genre):
    # Initialize session state for selected books
    books_selected =[]
    for genre in books_by_genre.keys():
        if genre:
            st.subheader(f"Genre: {genre}")
            # Create rows of books
            for i in range(0, len(books_by_genre[genre]), 3):  # Adjust the number per row as needed
                cols = st.columns(3)
                for col, book in zip(cols, books_by_genre[genre][i:i+3]):
                    with col:
                        st.write(f"Title: {book['title']}")
                        st.write(f"Author: {book['author']}")
                        # Check if the book is already selected
                        is_selected = book['_id'] in st.session_state['selected_books_ids']
                        if st.checkbox("Select book", value=is_selected, key=f"{book['_id']}, {genre}"):
                            # Add or remove the book based on checkbox state
                            if book['_id'] not in st.session_state['selected_books_ids']:
                                books_selected.append(book['_id'])
                            else:
                                books_selected.remove(book['_id'])
    return books_selected



def retrieve_books_by_ids(collection, book_ids):
    # Convert string IDs to ObjectIDs if necessary
    object_ids = [ObjectId(id) for id in book_ids]
    projection = {"_id": 1, "book_id": 1, "title": 1, "author": 1, 
                  "language": 1, "publisher": 1, "year_published": 1, "description": 1}
    books = collection.find({"_id": {"$in": object_ids}}, projection)
    return list(books)



def book_carousel_retrieved(books):
    # Create a list of items for the carousel
    links = []
    carousel_items = []
    for book in books:
        item = dict(title=str(book['title']),text=book['author'],img= "https://www.mobileread.com/forums/attachment.php?s=d9790f523ff1e127cf8e7160d8e3e671&attachmentid=111305&d=1378926764")
        carousel_items.append(item)
        links.append( book['book_id'])
    # Display the carousel in the Streamlit app
    return carousel_items,links


def main(_app):
    st.title("Welcome to the Book Recommendation App")
    books_selected= st.session_state['selected_books_ids']
    user_choice = st.radio("Choose your option:", ('Enter as Guest', 'Get Book Recommendations'))
    if user_choice == 'Get Book Recommendations' and st.session_state['user_preference']:
        user_specific_input = st.text_input("Searching for Book ?")
        if st.button("Search for books recomended: "):
            results = _app.query(user_specific_input, books_selected)
            book_ids = [docid for docid, _ in results]
            ranked_books= retrieve_books_by_ids(collection,book_ids )
            carousel_items, links= book_carousel_retrieved(ranked_books)
            i=0
            for elem in carousel_items:
                st.write("Book Title:", elem['title'], " - Author: ",elem['text'])
                st.write("https://www.goodreads.com/"+links[i]+ "\n\n")
                i+=1
            st.session_state['carousel_items']=carousel_items
        if st.session_state['carousel_items'] !=[]:
            carousel(items=st.session_state['carousel_items'], width=1)
            st.session_state['carousel_items']=[]
           
    elif user_choice == 'Get Book Recommendations':
        genres = retrieve_genres(collection)  # Fetch genres
        # Display genres for selection
        selected_genres = st.multiselect("Select your preferred genres", genres)
        if st.button("Submit Selection"):
            st.write(f"Select Your favorite Books : \n{selected_genres}")
        if len(selected_genres)<2 :
                st.write("Sorry you must select at least 2 cathegories!")
        else:
            st.subheader(f"Select the books you are more interrested by :")
            book_categories = retrieve_books(selected_genres, collection)
            books_selected = book_carousel_selection(book_categories)
            
        if st.button('Submit Final Selections'):
            # Now process the selected book ids
            st.write("Selected Book IDs:", books_selected)
        user_specific_input = st.text_input("Searching for Book ?")
        if st.button("Search for books recomended: "):
            results = _app.query(user_specific_input, books_selected)
            book_ids = [docid for docid, _ in results]
            ranked_books= retrieve_books_by_ids(collection,book_ids )
            carousel_items , links = book_carousel_retrieved(ranked_books)
            i=0
            for elem in carousel_items:
                st.write("Book Title:", elem['title'], " - Author: ",elem['text'])
                st.write("https://www.goodreads.com/"+links[i]+ "\n\n")
                i+=1
            st.session_state['carousel_items']=carousel_items
            #if carousel_items:
               # carousel(items=carousel_items, width=1)
               # st.session_state['user_preference'] = books_selected
                # Display books and let user select, then process selection
        if st.session_state['carousel_items'] !=[]:
            carousel(items=st.session_state['carousel_items'], width=1)
            st.session_state['carousel_items']=[]
            st.session_state['user_preference']= books_selected
            
    elif user_choice == 'Enter as Guest':
        #carousel_items=[]
        st.write("Welcome, guest! Enjoy the general content.")
        user_input = st.text_input("Searching for Book ?")
        if st.button("Search for books: "):
            results = _app.query(user_input)
            book_ids = [docid for docid, _ in results]
            ranked_books= retrieve_books_by_ids(collection,book_ids )
            carousel_items, links=book_carousel_retrieved(ranked_books)
            i=0
            for elem in carousel_items:
                st.write("Book Title:", elem['title'], " - Author: ",elem['text'])
                st.write("https://www.goodreads.com/"+links[i]+ "\n\n")
                i+=1

            st.session_state['carousel_items']=carousel_items
        if st.session_state['carousel_items'] !=[]:
            carousel(items=st.session_state['carousel_items'], width=1)
            st.session_state['carousel_items']=[]

# Run the app
if __name__ == "__main__":
    app=create_app()
    main(app)