import graphlab as gl

# local imports
from data_etl import (
      get_train_test
    , process_dataframe
    , create_submission
    )
from feature_engr import (
      TFIDFTransformer
    )

train, test = get_train_test()

# call wrapper method process_dataframe on train/test
train =  process_dataframe(train)
test = process_dataframe(test)

tfidf = TFIDFTransformer()
train = tfidf.fit_transform(train)
test = tfidf.transform(test)

# train logistic regression on training data with tf-idf as features and predict on testing data
train = train.dropna()
model = gl.logistic_classifier.create(train, target='sponsored', features=['tfidf'], class_weights='auto')

test = test.dropna()
ypred = model.predict(test)

# create submission.csv
create_submission(test, ypred)
