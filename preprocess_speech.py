from snowballstemmer import stemmer
from nltk.tokenize import word_tokenize
import string
import pandas as pd
from variables import stopwords
import pandas as pd


def preprocess_speech(text):
    greek_stemmer = stemmer('greek')
    translator = str.maketrans("", "", string.punctuation + "’")
    # Remove punctuation
    text = text.translate(translator)
    # print(text)
    # make lower case
    tokens = word_tokenize(text.lower())
    # print(tokens)
    # Remove numbers
    tokens = [word for word in tokens if word.isalpha()]
    # Stem only the words that are not in the stopwords list (εδώ ακόμα έχουν τόνους πριν το stemming)
    stem_words = [greek_stemmer.stemWord(token) for token in tokens if token not in stopwords]
    # print(stem_words)
    return ' '.join(stem_words)



def main():
    df = pd.read_csv("Greek_Parliament_Proceedings_1989_2020.csv", encoding='utf-8', nrows=3)
    print(df.head())
    # Preprocess the speech column and store back the results in the same column
    df['speech'] = df['speech'].apply(preprocess_speech)
    print(df.head())


if __name__ == '__main__':
    main()