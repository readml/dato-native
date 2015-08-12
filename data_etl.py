"""data_etl.py

Extract, Transform, Load data into the proper format
"""
import re
from urlparse import urlparse
import graphlab as gl
from config import PATH_TO_JSON, PATH_TO_TRAIN_LABELS, PATH_TO_TEST_LABELS

def get_train_test():
    # read json blocks from path PATH_TO_JSON
    sf = gl.SFrame.read_csv(PATH_TO_JSON, header=False)
    sf = sf.unpack('X1',column_name_prefix='')

    # read train and test labels from paths PATH_TO_TRAIN_LABELS and PATH_TO_TEST_LABELS
    train_labels = gl.SFrame.read_csv(PATH_TO_TRAIN_LABELS)
    test_labels = gl.SFrame.read_csv(PATH_TO_TEST_LABELS)

    # create a new columns "id" from parsing urlId and drop file columns
    train_labels['id'] = train_labels['file'].apply(lambda x: str(x.split('_')[0] ))
    train_labels = train_labels.remove_column('file')
    test_labels['id'] = test_labels['file'].apply(lambda x: str(x.split('_')[0] ))
    test_labels = test_labels.remove_column('file')

    # join labels with html data from training and testing SFrames
    train = train_labels.join(sf, on='id', how='left')
    test = test_labels.join(sf, on='id', how='left')

    return train, test

def create_count_features(sf):
    """ a simple method to create some basic features on an SFrame """
    sf['num_images'] = sf['images'].apply(lambda x: len(x))
    sf['num_links'] = sf['links'].apply(lambda x: len(x))
    sf['num_clean_chars'] = sf['text_clean'].apply(lambda x: len(x))
    return sf

def clean_text(sf):
    """ a simple method to clean the text within an html response """
    sf['text_clean'] = sf['text'].apply(lambda x:
        re.sub(r'[\n\t,.:;()\-\/]+', ' ', ' '.join(x)))
    sf['text_clean'] = sf['text_clean'].apply(lambda x: re.sub(r'\s{2,}', ' ', x))
    sf['text_clean'] = sf['text_clean'].apply(lambda x: x.strip())
    return sf

def create_link_netloc(sf):
    """ a simple method to extract the net(work) loc(ations) in url links """
    def get_netloc(link):
        try:
            return urlparse(link).netloc
        except:
            return ''

    sf["netlocs"] = sf["links"].apply(
        lambda x: ' '.join([get_netloc(link) for link in x])
    ) 
    return sf

def process_dataframe(sf):
    """ a wrapper method around the 2 methods above """
    sf = clean_text(sf)
    sf = create_count_features(sf)
    sf = create_link_netloc(sf)
    return sf

def create_submission(test, ypred):
    """ create submission.csv """
    submission = gl.SFrame()
    submission['sponsored'] = ypred 
    submission['file'] = test['id'].apply(lambda x: x + '_raw_html.txt')
    submission.save('submission_version_1.csv', format='csv')
