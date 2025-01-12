from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle
from scipy.sparse.linalg import svds
from preprocess_speech import preprocess_speech
from load_file import load_processed_file
from LSI import LSI


def all_speeches_of_member(data: pd.DataFrame, member_name: int):
    '''
    Get all speeches of a member.
    '''
    return data[data['member_name'] == member_name]


def main():
    df = load_processed_file(rows=250)
    df = df[['member_name', 'speech']]

    # Group by member
    grouped = df.groupby('member_name')
    # For each member, get all speeches in a single string
    all_speeches = pd.DataFrame(columns=['member_name', 'speech'])
    for name, group in grouped:
        all_speeches = pd.concat([all_speeches, pd.DataFrame([{'member_name': name, 'speech': ' '.join(group['speech'])}])], ignore_index=True)

    print(all_speeches.head(-1))

    # Create LSI object and reduce the dimensionality of the concatenated speeches
    lsi = LSI.load_object('lsi.pkl')

    # Reduce the dimensionality of the speeches to the consepts space
    speeches_reduced = []
    for speech in all_speeches['speech']:
        speeches_reduced.append(lsi.lsi_vectorize(speech))
    all_speeches['speech'] = speeches_reduced

    print(all_speeches.head(-1))


if __name__ == '__main__':
    main()




    




if __name__ == '__main__':
    main()