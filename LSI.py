from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle
from scipy.sparse.linalg import svds
from preprocess_speech import preprocess_speech
from load_file import load_processed_file


class LSI:
    '''
    A class for tf-idf vectorizer objects and the matrixes of SVD.
    '''
    def __init__(self, data: pd.DataFrame):    
        # Calculate TF-IDF for each group
        self.vectorizer: TfidfVectorizer = TfidfVectorizer()
        print('Fitting the TfidfVectorizer object...')
        self.vectorizer.fit(data['speech'])
        print('Transforming the data...')
        self.tf_idf_matrix:  np.ndarray = self.vectorizer.transform(data['speech'])
        
        # SVD
        print('Calculating the SVD...')
        self.U, self.S, self.Vt = svds(self.tf_idf_matrix, k=200, which='LM')  
        # Η σημαντικότητα των conspets είναι σε αύσουσα (??) σειρά στον S άρα οι τελευταίες στήλες του AV.T είναι οι σημαντικότερες
        
        
        
        # ΔΕΝ ΤΟ ΚΡΑΤΑΩ ΓΙΑ ΝΑ ΜΗΝ ΑΠΟΘΗΚΕΥΤΕΙ
        # Ομιλίες ως διάνυσμα σε κάποιον πολυδιάστατο χώρο (μειωμένης διάστασης) των σημαντικότερων θεματικών εννοιών
        # self.speeches_reduced_to_concepts = np.dot(self.U, np.diag(self.S))

        # Pandas dataframe with th id of the speech and the reduced speech vector
        # self.speeches_reduced_to_concepts_df = pd.DataFrame(self.speeches_reduced_to_concepts)
        # self.speeches_reduced_to_concepts_df['id'] = data['id']

    
    def tf_idf_vectorize(self, text: str):
        '''
        Vectorize a text using the fitted TfidfVectorizer object.
        '''
        return self.vectorizer.transform([text])
    

    def lsi_vectorize(self, text: str):
        '''
        Vectorize text and reduce its dimension using the precomputed LSI model.
        '''
        return np.dot(self.tf_idf_vectorize(text).toarray(), self.Vt.T).flatten()
    

    def save_object(self, path: str):
        '''
        Save the LSI object to a binary file.
        '''
        with open(path, 'wb') as file:
            pickle.dump(self, file)


    @staticmethod
    def load_object(path: str):
        '''
        Load an LSI object from a binary file.
        '''
        with open(path, 'rb') as file:
            return pickle.load(file)
        
    



def main():
    # Load the processed data
    data: pd.DataFrame = load_processed_file()

    # Create the LSI object
    lsi = LSI(data)
    print('LSI object created.')
    

    # -----------------Test the LSI object---------------
    test = False
    if test:
        print(lsi.speeches_reduced_to_concepts)
        sample_text = 'Καλημέρα σας αγαπητοι συναδερφοι δημοκρατια σημαντική'
        sample_text_processed = preprocess_speech(sample_text)
        print('Sample text reduced to concepts:')
        print(lsi.lsi_vectorize(sample_text_processed))

        print(lsi.speeches_reduced_to_concepts_df['id'])


    # ------------------SAVE OUTPUT-------------------
    print('Saving the LSI object...')
    lsi.save_object('lsi.pkl')
    print('LSI object saved to lsi.pkl')


    # This was created and then rerun without it so as to not create a huge lsi.pkl file
    # df_speeches = lsi.speeches_reduced_to_concepts_df
    # # Save this to a csv file
    # df_speeches.to_csv('speeches_reduced_to_concepts.csv')



if __name__ == '__main__':
    main()
        

