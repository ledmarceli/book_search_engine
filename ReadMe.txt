This is the search engine implementation for the book summary dataset by group 1 for ECS736U/ECS736P - Information Retrieval - 2023/24.
Members of the group and their responsibilities are:
Marceli Franciszek Ciesielski (Ingestion, Query Processing, File organisation)
Iliass El Yaakoubi Benssaleh (Evaluation)
Kem Lenny Ibodi (Ranking)
Bartosz Michal Watrobinski (Indexing)

Required Installations:
python 3.10
nltk
scikit-learn
pandas

Instructions to run:
1. Ensure you have installed all the required libraries.
2. Download the booksummaries.txt and place it in this directory. Download from from: https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset?resource=download
3. Run the indexing.py file first. It might take a few minutes to process.
4. To run the search engine for your own query run main.py and follow the instructions.
5. To run the evaluation procedure, run the evaluation.py.

Description of files:
booksummaries.txt - contains the original dataset.
indexing.py - contains all the preprocessing and indexing code. Run this code to reobtain the processed engine files in processed_files folder.
save_and_read_files.py - contains functions for saving and reading preprocessed datasets and index files in processed_files folder.
ranking.py - contains all the ranking functions.
process_query.py - contains all the preprocessing for a user query.
main.py - run this to use the search engine.
evaluation.py - run this to conduct an evaluation of the search engine on the test dataset.