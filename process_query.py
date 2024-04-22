import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_query(query):
    # Tokenize the query
    tokens = word_tokenize(query)
    # Convert tokens to lowercase
    tokens = [token.lower() for token in tokens]
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # Stemming
    tokens = [stemmer.stem(token) for token in tokens]
    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens