import pickle
from collections import defaultdict
import csv

def save_tfidf_matrices(file_path, tfidf_matrices):
    with open(file_path, 'wb') as file:
        pickle.dump(tfidf_matrices, file)


def save_inverted_indexes(inverted_indexes_file, inverted_indexes):
    with open(inverted_indexes_file, 'w', encoding='utf-8') as file:
        for field, inverted_index in inverted_indexes.items():
            file.write(f"Field: {field}\n")
            for term, postings_list in inverted_index.items():
                # Write the term and postings list directly without encoding
                file.write(f"{term}: {postings_list}\n")
            file.write("\n")


def load_tfidf_matrices(file_path):
    with open(file_path, 'rb') as file:
        tfidf_matrices = pickle.load(file)
    return tfidf_matrices


def load_inverted_indexes(file_path):
    inverted_indexes = {}
    current_field = None
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespaces
            if line.startswith("Field: "):
                current_field = line.replace("Field: ", "")
                inverted_indexes[current_field] = defaultdict(list)
            elif line:  # Non-empty line
                # Split the line into term and postings list string
                term, postings_list_str = line.split(": ", 1)
                # Convert postings list string to list
                postings_list = eval(postings_list_str)
                # Store the term and postings list in the inverted index
                inverted_indexes[current_field][term] = postings_list
    return inverted_indexes


def load_test(csv_file_path):
    queries = []
    relevant_documents = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            queries.append(row[0])  # Assuming query is in the first column
            ids_with_spaces = row[1].split(',')
            ids_stripped = [id.strip() for id in ids_with_spaces]  # Remove leading spaces
            relevant_documents.append(ids_stripped)
    return queries, relevant_documents