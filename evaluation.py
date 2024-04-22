print("Loading the search engine:")
from process_query import preprocess_query
from ranking import BM25_ranker, display_rank
from save_and_read_files import load_inverted_indexes, load_tfidf_matrices, load_test
import pandas as pd
import os
import time

os.system('cls' if os.name=='nt' else 'clear')
print("Loading the search engine:")

df = pd.read_csv("processed_files/preprocessed_dataframe.csv", header=0)
processed_dataframe = pd.read_csv("processed_files/processed_dataframe.csv", header=0)
inverted_indexes = load_inverted_indexes("processed_files/inverted_indexes.txt")
tfidf_matrices = load_tfidf_matrices("processed_files/tfidf_matrices.pkl")
queries, relevant_documents = load_test("testdata.csv")

def precision_at_k(query, relevant_documents, k=3):
    """Calculate precision at k for a given query."""
    ranked_docs = BM25_ranker(query, processed_dataframe, inverted_indexes, tfidf_matrices, df)
    retrieved = [str(doc_id) for doc_id, _ in ranked_docs[:k]]  # Convert document IDs to strings

    true_positives = len(set(retrieved) & set(relevant_documents))
    false_positives = len(retrieved) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    return precision

def recall_at_k(query, relevant_documents, k=3):
    """Calculate recall at k for a given query."""
    ranked_docs = BM25_ranker(query, processed_dataframe, inverted_indexes, tfidf_matrices, df)
    retrieved = [str(doc_id) for doc_id, _ in ranked_docs[:k]]  # Convert document IDs to strings

    true_positives = len(set(retrieved) & set(relevant_documents))

    recall = true_positives / len(relevant_documents) if len(relevant_documents) > 0 else 0

    return recall

def f1_score_at_k(query, relevant_documents, k=3):
    """Calculate F1 score at k for a given query."""
    precision = precision_at_k(query, relevant_documents, k)
    recall = recall_at_k(query, relevant_documents, k)

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score

def reciprocal_rank(relevant_documents, ranked_documents):
    for rank, doc_id in enumerate(ranked_documents, 1):
        if doc_id in relevant_documents:
            return 1 / rank
    return 0  # If no relevant documents are retrieved, return 0

def mean_reciprocal_rank(queries, relevance_data):
    total_rr = 0
    num_queries = len(queries)

    for query, relevant_documents in zip(queries, relevance_data):
        ranked_docs = BM25_ranker(query, processed_dataframe, inverted_indexes, tfidf_matrices, df)
        retrieved = [str(doc_id) for doc_id, _ in ranked_docs]
        rr = reciprocal_rank(relevant_documents, retrieved)
        total_rr += rr

    mrr = total_rr / num_queries if num_queries > 0 else 0
    return mrr

def query_latency(query, documents):
    """Calculate the query latency for a given query."""
    start_time = time.time()
    ranked_docs = BM25_ranker(query, documents, inverted_indexes, tfidf_matrices, df)
    end_time = time.time()
    latency = end_time - start_time
    return latency

print("Evaluating...")

precisions = []  # List to store precision values
# Test precision at k=5 for each query
for query, relevant_docs in zip(queries, relevant_documents):
    precision = precision_at_k(query, relevant_docs, k=3)
    precisions.append(precision)
average_precision = sum(precisions) / len(precisions)

print("Average Precision:", average_precision)

# Test recall at k=5 for each query
recalls = []  # List to store recall values
for query, relevant_docs in zip(queries, relevant_documents):
    recall = recall_at_k(query, relevant_docs, k=3)
    recalls.append(recall)

# Calculate average recall
average_recall = sum(recalls) / len(recalls)
print("Average Recall:", average_recall)

# Test precision, recall, and F1 score at k=5 for each query
f1_scores = []  # List to store F1 score values
for query, relevant_docs in zip(queries, relevant_documents):
    f1_score = f1_score_at_k(query, relevant_docs, k=3)
    f1_scores.append(f1_score)

# Calculate average F1 score
average_f1_score = sum(f1_scores) / len(f1_scores)
print("Average F1 Score:", average_f1_score)

# Test MRR for the set of queries
mrr = mean_reciprocal_rank(queries, relevant_documents)
print(f'Mean Reciprocal Rank for queries:', mrr)

# Test query latency for each query
latencies = []  # List to store latency values
for query in queries:
    latency = query_latency(query, processed_dataframe)
    latencies.append(latency)

# Calculate average latency
average_latency = sum(latencies) / len(latencies)
print("Average Query Latency:", average_latency, "seconds")

print("Finished Evaluating.")