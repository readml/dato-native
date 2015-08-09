import graphlab as gl

# local imports
from data_etl import (
      get_train_test
    , process_dataframe
    )
from feature_engr import (
      GloveTransformer
    )

train, test = get_train_test()

# call wrapper method process_dataframe on train/test
train =  process_dataframe(train)
test = process_dataframe(test)

# Continue Bag Of Word Text Vectors
glovet = GloveTransformer("glove.6B.%sd.txt.gz" % 300)
train['glove'] = train['text_clean'].apply(lambda x: glovet.txt2avg_vector(x))
test['glove'] = test['text_clean'].apply(lambda x: glovet.txt2avg_vector(x))

# train logistic regression on training data with tf-idf as features and predict on testing data
train = train.dropna()
model = gl.logistic_classifier.create(train, target='sponsored', features=['glove'], class_weights='auto')

test = test.dropna()
model.evaluate(test)
