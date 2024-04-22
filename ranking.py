import math
from collections import defaultdict
import pandas as pd
from process_query import preprocess_query

column_names = ['wikipedia_id', 'freebase_id', 'title', 'author', 'publication_date', 'genres', "summary"]


def calculate_bm25_score(n, f, tf, r, N, dl, avdl, k1=1.5, b=0.75):
  # Calculate BM25 score for a single term in a document.
  K = k1 * ((1 - b) + b * (float(dl) / float(avdl)))
  idf = math.log((N - n + 0.5) / (n + 0.5) + 1)
  score = idf * ((f * (k1 + 1)) / (f + K)) * ((tf * (1.2 + 1)) / (tf + 1.2))
  return score

def rank_documents(query_terms, inverted_indexes, tfidf_matrices, N, avdl, k1=1.5, b=0.75, ):

  # Create dictionary to store bm25 scores
  document_scores = defaultdict(float)

  # for each term in the query get the bm25 score for each field it occurs in
  # then accumulate the scores and return a dictionary with scores for each document
  for term in query_terms:
    count = 1
    for field, inverted_index in inverted_indexes.items():
      if term in inverted_index:
        df = len(inverted_index[term])
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        for doc_index, tfidf_score in inverted_index[term]:
          f = tfidf_score
          tf = query_terms.count(term)
          dl = len(tfidf_matrices[field][doc_index].toarray()[0])
          score = calculate_bm25_score(df, f, tf, 0, N, dl, avdl, k1, b)
          document_scores[doc_index] += score

  return document_scores

def BM25_ranker(query, documents, inverted_indexes, tfidf_matrices, df):

  # Preprocess the query
  preprocessed_query = preprocess_query(query)

  # Calculate average document length
  total_documents = len(df)
  total_terms_in_collection = sum(len(documents[field]) for field in documents)
  average_document_length = total_terms_in_collection / total_documents

  # Rank documents based on the query
  document_scores = rank_documents(preprocessed_query, inverted_indexes, tfidf_matrices, total_documents, average_document_length)

  # Sort documents based on their BM25 scores
  sorted_documents = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)

  # Return documents sorted by bm25 score
  return sorted_documents


def display_rank(top_n, sorted_documents, df):
    # Display top N ranked documents for a given sorted document list
    for doc_index, score in sorted_documents[:top_n]:
        print(f"Document Index: {doc_index}, BM25 Score: {score}")
        print("Title:", df.iloc[doc_index]['title'])
        print("Author:", df.iloc[doc_index]['author'])
        print("Genres:", df.iloc[doc_index]['genres'])
        print("Summary:", df.iloc[doc_index]['summary'])
        print("--------------------------------------")