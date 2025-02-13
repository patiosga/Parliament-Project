import pickle
import pandas as pd
from collections import defaultdict
from LSA import LSA

# Load the processed data (CSV)
def load_processed_file(rows=-1):
    file_path = r"processed_data.csv"
    if rows == -1:
        data = pd.read_csv(file_path, encoding='utf-8')
    else:
        data = pd.read_csv(file_path, encoding='utf-8', nrows=rows)
    return data

# Load the LSI model from a pickle file
def load_lsa_model(model_path='lsa.pkl'):
    with open(model_path, 'rb') as f:
        lsi_obj = pickle.load(f)
    return lsi_obj

# Building the inverted index with deduplication (store max weight)
def build_inverted_index(lsi_obj, processed_data):
    inverted_index = defaultdict(dict)
    
    # Loop through each document in the processed data
    for doc_id, speech in enumerate(processed_data['speech']):
        # Split the speech into words
        words = speech.split()

        # For each word in the speech, find the weight using the LSI model
        for word in words:
            if word in lsi_obj.vectorizer.vocabulary_:
                # Get the index of the word in the vocabulary
                word_index = lsi_obj.vectorizer.vocabulary_[word]
                # Get the TF-IDF weight of the word in the document
                weight = lsi_obj.tf_idf_matrix[doc_id, word_index]
                
                # Add the word and its weight to the inverted index, only keeping max weight per word-doc pair
                if word not in inverted_index or weight > inverted_index[word].get(doc_id, 0):
                    inverted_index[word][doc_id] = weight
    
    return inverted_index

# Save the inverted index to a pickle file
def save_inverted_index_to_pickle(inverted_index, filename='inverted_index.pkl'):
    # Prepare the data for saving as pickle file
    inverted_index_data = []
    for word, doc_weights in inverted_index.items():
        for doc_id, weight in doc_weights.items():
            inverted_index_data.append([word, doc_id, weight])
    
    # Convert to DataFrame
    df = pd.DataFrame(inverted_index_data, columns=['Word', 'Document ID', 'Weight'])
    
    # Create a dictionary to store the word occurrences per document
    inverted_index = defaultdict(list)
    for index, row in df.iterrows():
        inverted_index[row['Word']].append((row['Document ID'], row['Weight']))  # TF-IDF weight

    # Save the inverted index to a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(inverted_index, f)

if __name__ == '__main__':
    # Load your data and LSΑ model
    processed_data = load_processed_file()  # You can adjust the rows argument as needed
    lsi_obj = load_lsa_model()

    # Build the inverted index with deduplication
    inverted_index = build_inverted_index(lsi_obj, processed_data)

    # Save the inverted index to a CSV file
    save_inverted_index_to_pickle(inverted_index, filename='inverted_index.csv')

    print("Inverted index saved to 'inverted_index.csv'.")
