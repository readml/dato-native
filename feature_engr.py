"""feature_engr.py

Creating the good features to be used in the SFrame
"""
# common python
import os
import csv
import pandas as pd
import numpy as np

# graphlab
import graphlab as gl
from graphlab.toolkits.feature_engineering import TFIDF

# local imports
from config import GLOVE_FOLDER

class TFIDFTransformer(object):
    """ Wrapper around GraphLabs bag of words and TFIDF models to act
    more like a scikit estimator """

    def __init__(self, column_name):
        self.column_name = column_name

    def fit_transform(self, train):
        """ unlike scikit, features and targets are encapsulated in the 
        train sframe
        
        Parameters
        ----------
        train: train sframe, with column 'text_clean'

        Returns
        -------
        train: TFIDF transformed version of train, with the features located 
            in column 'tfidf'.
        """
        # create word counts and remove countwords
        bow_trn = gl.text_analytics.count_words(train[self.column_name])
        bow_trn = bow_trn.dict_trim_by_keys(gl.text_analytics.stopwords())
        
        # add bag of words to sframe
        train['bow_'+self.column_name] = bow_trn
        
        self.encoder = gl.feature_engineering.create(
              train
            , TFIDF('bow_'+self.column_name, output_column_name='tfidf_'+self.column_name)
            )
        
        return self.encoder.transform(train)

    def transform(self, test):
        """
        Parameters
        ----------
        test: train sframe, with column 'text_clean'

        Returns
        -------
        train: TFIDF transformed version of train, with the features located 
            in column 'tfidf'.
        """
        # create word counts and remove countwords
        bow_tst = gl.text_analytics.count_words(test[self.column_name])
        bow_tst = bow_tst.dict_trim_by_keys(gl.text_analytics.stopwords())

        # add the bag of words to both sframes
        test['bow_'+self.column_name] = bow_tst

        return self.encoder.transform(test)

class GloveTransformer(object):

    def __init__(self, glove_file, nrows=50000):
        data_path = os.path.join(GLOVE_FOLDER, glove_file)
        raw = pd.read_csv(data_path, header=None, sep=' ', quoting=csv.QUOTE_NONE, nrows=nrows)
        keys = raw[0].values
        self.vectors = raw[range(1, len(raw.columns))].values
        self.vector_dim = self.vectors.shape[1]

        # lookup will have (key, val) -> (word-string, row index in self.vectors)
        row_indexes = range(self.vectors.shape[0])
        self.lookup = dict(zip(keys, row_indexes))
        self.reverse_lookup = dict(zip(row_indexes, keys))

    def txt2vectors(self, txt):
        """ Calculate the list of vector representations for words in txt """
        words = txt.split(' ')
        words = [word.lower() for word in words]
        word_seq_vect = []
        for word in words:
            if word in self.lookup:
                word_seq_vect.append(self.vectors[self.lookup[word]])
            else:
                # random values for UNK tokens
                word_seq_vect.append(np.random.rand(self.vector_dim))
        word_seq_vect = np.vstack(word_seq_vect)
        return word_seq_vect

    def txt2avg_vector(self, txt):
        """ Calculate the average vector representation of the input text """
        vectors = self.txt2vectors(txt)
        avg_vector = np.mean(vectors, axis=0)
        avg_vector = np.nan_to_num(avg_vector)
        return avg_vector
