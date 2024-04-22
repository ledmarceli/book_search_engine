print("Loading the search engine:")
from process_query import preprocess_query
from ranking import BM25_ranker, display_rank
from save_and_read_files import load_inverted_indexes, load_tfidf_matrices
import pandas as pd
import os

os.system('cls' if os.name=='nt' else 'clear')
print("Loading the search engine:")


df = pd.read_csv("processed_files/preprocessed_dataframe.csv", header=0)
processed_dataframe = pd.read_csv("processed_files/processed_dataframe.csv", header=0)
inverted_indexes = load_inverted_indexes("processed_files/inverted_indexes.txt")
tfidf_matrices = load_tfidf_matrices("processed_files/tfidf_matrices.pkl")



print("Welcome to the book summmary search engine!")
running = True
while(running):
    user_query = input("Type in your query, or type exit if you want to close the program.")
    if not user_query == "exit":
        test = BM25_ranker(user_query, processed_dataframe, inverted_indexes, tfidf_matrices, df)
        display_rank(5, test, df)
    else:
        running = False