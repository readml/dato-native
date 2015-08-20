import graphlab as gl

# local imports
from data_etl import (
      get_train_test
    , process_dataframe
    , create_balanced_validation
    , create_submission
    )
from feature_engr import (
      TFIDFTransformer
    )

train, test = get_train_test()

# call wrapper method process_dataframe on train/test
train = process_dataframe(train)
test = process_dataframe(test)

# tfidf on text_clean
tfidf = TFIDFTransformer('text_clean')
train = tfidf.fit_transform(train)
test = tfidf.transform(test)

# tfidf on network locations
tfidf = TFIDFTransformer('netlocs')
train = tfidf.fit_transform(train)
test = tfidf.transform(test)

# train logistic regression on training data with tf-idf as features and predict on testing data
train = train.dropna()

valid = create_balanced_validation(train, n_positive_examples=500)

model = gl.logistic_classifier.create(
    train,
    target='sponsored',
    features=['tfidf_text_clean', 'tfidf_netlocs'],
    l2_penalty=1e3,
    class_weights='auto',
    validation_set=valid
)

test = test.dropna()
ypred = model.predict(test)

create_submission(test, ypred, name="data/submission_version_1.csv")
