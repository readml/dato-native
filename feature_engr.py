"""feature_engr.py

Creating the good features to be used in the SFrame
"""
import graphlab as gl
from graphlab.toolkits.feature_engineering import TFIDF

class TFIDFTransformer(object):
    """ Wrapper around GraphLabs bag of words and TFIDF models to act
    more like a scikit estimator """

    def __init__(self):
        pass

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
        bow_trn = gl.text_analytics.count_words(train['text_clean'])
        bow_trn = bow_trn.dict_trim_by_keys(gl.text_analytics.stopwords())
        
        # add bag of words to sframe
        train['bow'] = bow_trn
        
        self.encoder = gl.feature_engineering.create(
              train
            , TFIDF('bow', output_column_name='tfidf')
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
        bow_tst = gl.text_analytics.count_words(test['text_clean'])
        bow_tst = bow_tst.dict_trim_by_keys(gl.text_analytics.stopwords())

        # add the bag of words to both sframes
        test['bow'] = bow_tst

        return self.encoder.transform(test)
