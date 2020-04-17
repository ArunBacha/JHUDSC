import  pandas as pd
import  numpy as np
from pymongo import MongoClient
#Inserting  data into MongoDB
def main():
    #loading datasets 
    book_tags = pd.read_csv("book_tags.csv")
    books = pd.read_csv("books.csv")    
    ratings = pd.read_csv("ratings.csv")    
    tags = pd.read_csv("tags.csv")  
    books_to_read = pd.read_csv("to_read.csv")
    #MongoDB Details: 
    MONGODB_HOST = 'localhost'
    MONGODB_PORT = 27017
    DB_NAME = 'BigData_Project'
    Book_Tags = 'Book_Tags'
    Books = "Books"
    Ratings = "Ratings"
    Tags = "Tags"
    Books_To_Read = "Books_To_Read"
    
    #Setting Up Connection to MongoDB
    connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
    collection_Book_Tags = connection[DB_NAME][Book_Tags]
    print("Connection with Book_Tags")
    collection_Books = connection[DB_NAME][Books]
    collection_Ratings = connection[DB_NAME][Ratings]
    collection_Tags = connection[DB_NAME][Tags]
    collection_Books_To_Read = connection[DB_NAME][Books_To_Read]
    
    #Inserting data into MongoDB
    print("MongoDb insertion")
    collection_Book_Tags.insert_many(book_tags.to_dict('records'))
    print("Data inserted into Book_Tags collection \n")
    collection_Books.insert_many(books.to_dict('records'))
    print("Data inserted into Books collection \n")
    collection_Ratings.insert_many(ratings.to_dict('records'))
    print("Data inserted into Ratings collection \n")
    collection_Tags.insert_many(tags.to_dict('records'))
    print("Data inserted into tags collection \n")
    collection_Books_To_Read.insert_many(books_to_read.to_dict('records'))
    print("Data inserted into books to read collection \n")
       
    
if __name__=="__main__":
    main()   



