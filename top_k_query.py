import pandas as pd
import math
from collections import defaultdict
from preprocess_speech import preprocess_speech
import pickle
import numpy as np
from LSA import LSA
import time
import heapq


def keep_top_k(doc_weights, k):
    """Keeps only the top k highest weighted documents."""
    return heapq.nlargest(k, doc_weights.items(), key=lambda x: x[1])


# Function to perform top-k query search
def top_k_query(query_terms, k=20, inverted_index=None):

    # Load inverted index 
    with open('inverted_index.pkl', 'rb') as f:
        inverted_index = pickle.load(f)
    # Load corresponding doc ids
    ids = np.load('processed_ids.npy')


    Σ = {}  # Store similarity scores per document

    # Compute similarity score for each document
    for t in query_terms:
        if t in inverted_index:
            # print(f"Term {t} found in the inverted index.")
            for doc_id, weight in inverted_index[t]:  # TF-IDF weight from pickle file
                if doc_id not in Σ:
                    Σ[doc_id] = weight
                else:
                    Σ[doc_id] += weight  # Sum of TF-IDF scores

    # Normalize by document vector length
    # for doc_id in Σ:
    #     if document_vector_lengths[doc_id] > 0:
    #         Σ[doc_id] /= document_vector_lengths[doc_id]  # Cosine similarity normalization
    #!!!!! This is not needed after all because the tf idf vectorizer already normalizes the vectors to have norm 1

    top_k_docs = keep_top_k(Σ, k)  # Keep only top-k documents

    final_ids = []
    for doc_id, score in top_k_docs:
        final_ids.append(int(ids[doc_id]))

    return final_ids  # Return top-k results



if __name__ == "__main__":
    time1 = time.time()
    # Load the inverted index from a pickle file
    with open('inverted_index.pkl', 'rb') as f:
        inverted_index = pickle.load(f)
    print('Inverted index loaded.')
    time2 = time.time()
    query = True
    if query:
        # Example query
        query_terms = 'πασοκ'  # Example query terms

        query_terms = preprocess_speech(query_terms).split()
        print(f"Query terms: {query_terms}")
        top_k_documents = top_k_query(query_terms)

        # Display top-k results
        print("\nFinal Top-K Documents:")
        for doc_id in top_k_documents:
            print(f"Document ID: {doc_id}")
    time3 = time.time()
    print(f"Time to load inverted index: {time2-time1}")
    print(f"Time to perform query: {time3-time2}")
    print(f"Total time: {time3-time1}")
