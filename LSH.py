import numpy as np
from simhash import Simhash
from sklearn.preprocessing import normalize
from LSI import LSI
import pickle
from load_file import load_processed_file


class LSH:



# Get the weight of each consept in the S matrix
concept_weights = []

# Simulated data: n = 1000 objects, 200 features
n = 1000
features = np.random.randn(n, 200)

# Function to compute the SimHash for a vector with concept weights
def compute_weighted_simhash(concept_vector, concept_weights):
    return Simhash(np.dot(concept_vector, concept_weights).flatten())


# Compute all the simhashes for the objects
def all_simhashes(speeches, concept_weights):
    return [compute_weighted_simhash(speech, concept_weights) for speech in speeches]


simhashes = all_simhashes(features, concept_weights)
# Create a dictionary to store hashes and corresponding indices
hash_buckets = {}

# Insert the simhashes into the buckets
for idx, simhash in enumerate(simhashes):
    h = simhash.value  # SimHash value as the key
    if h not in hash_buckets:
        hash_buckets[h] = []
    hash_buckets[h].append(idx)




# Query: Find top k similar items to the first object
query = features[0]
query_simhash = compute_weighted_simhash(query, concept_weights)

# Find candidates by looking for similar hash values
k = 5
similar_items = []

# Retrieve similar objects based on hash values
for h in hash_buckets:
    # SimHash similarity is based on the number of differing bits
    if bin(h ^ query_simhash.value).count('1') <= 3:  # We allow a small number of bit differences
        similar_items.extend(hash_buckets[h])

# Output the results
print(f"Top {k} similar objects to the query:")
for idx in similar_items[:k]:
    print(f"Index: {idx}")
