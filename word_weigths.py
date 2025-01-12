import pickle
from LSI import LSI


def get_word_weights_dict() -> dict:
    # Read the lsi.pkl file
    with open('lsi.pkl', 'rb') as file:
        lsi: LSI = pickle.load(file)
    
    # Get the weight of each word in the tf-idf matrix
    word_weights = lsi.vectorizer.idf_

    # Get the corresponding words
    words = lsi.vectorizer.get_feature_names_out()
    
    # Create a dictionary with the words as keys and their weights as values
    word_weights_dict = {}
    for word, weight in zip(words, word_weights):
        word_weights_dict[word] = weight

    print(lsi.S)

    return word_weights_dict



if __name__ == '__main__':
    # for word, weight in get_word_weights_dict().items():
    #     print(f'{word}: {weight}')

    get_word_weights_dict()