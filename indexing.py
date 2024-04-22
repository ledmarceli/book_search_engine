#imports
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from save_and_read_files import save_tfidf_matrices, save_inverted_indexes
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


#parameters
file_path = "booksummaries.txt"
column_names = ['wikipedia_id', 'freebase_id', 'title', 'author', 'publication_date', 'genres', "summary"]

#functions
def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text)
    # Convert to lowercase
    tokens = [token.lower() for token in tokens]
    # Remove stopwords
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    # Stemming
    tokens = [stemmer.stem(token) for token in tokens]
    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens back into a single string (MAY WANT TO GET RID OF THIS)
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

#convert genres into a list
def string_dict_to_string(string_dict):
    if string_dict=="" or pd.isna(string_dict):
        return "None"
    else:
        dict_obj = eval(string_dict)
        values_list = list(dict_obj.values())
        string = ', '.join(values_list)
        return string


#read and prepare the data
print("Reading and preparing data")
df = pd.read_csv(file_path, sep='\t', names=column_names)
df['genres'] = df['genres'].apply(string_dict_to_string)
df = df.fillna("None")
df.to_csv('processed_files/preprocessed_dataframe.csv', index=False)

#preprocess the dataframe
columns_to_preprocess = ["genres", "author", "title","summary"]
processed_dataframe = df.copy()  # Make a copy of the original dataframe
print("Preprocessing all the coulumns - this could take a few minutes.")
for column in columns_to_preprocess:
    processed_dataframe[column] = df[column].apply(preprocess_text)

print("Saving the processed dataframe")
processed_dataframe.to_csv('processed_files/processed_dataframe.csv', index=False)

# Defining separate vectorizers for each column
vectorizers = {
    'genres': TfidfVectorizer(tokenizer=lambda x: x.split(), preprocessor=lambda x: x, lowercase=False),
    'author': TfidfVectorizer(tokenizer=lambda x: x.split(), preprocessor=lambda x: x, lowercase=False),
    'title': TfidfVectorizer(tokenizer=lambda x: x.split(), preprocessor=lambda x: x, lowercase=False),
    'summary': TfidfVectorizer(tokenizer=lambda x: x.split(), preprocessor=lambda x: x, lowercase=False)
}

# using multiple fields
tfidf_matrices = {}
feature_names = {}
for field, vectorizer in vectorizers.items():
    tfidf_matrices[field] = vectorizer.fit_transform(processed_dataframe[field])
    feature_names[field] = vectorizer.get_feature_names_out()

# creating separate inverted indexes
inverted_indexes = {field: defaultdict(list) for field in vectorizers}

print("Indexing - this can take a few minutes")
# Filling each inverted index with TF-IDF values
for field in vectorizers:
    for doc_index, term_indices in enumerate(tfidf_matrices[field]):
        for term_index in term_indices.indices:
            term = feature_names[field][term_index]
            tfidf_score = tfidf_matrices[field][doc_index, term_index]
            inverted_indexes[field][term].append((doc_index, tfidf_score))


print("Saving")

inverted_indexes_file = "processed_files/inverted_indexes.txt"
file_path = "processed_files/tfidf_matrices.pkl"

save_tfidf_matrices(file_path, tfidf_matrices)
save_inverted_indexes(inverted_indexes_file, inverted_indexes)

print("Finished. The indexes have been saved in inverted_indexes.txt and the tfidf matrices have been saved to tfidf_matrices.pkl")