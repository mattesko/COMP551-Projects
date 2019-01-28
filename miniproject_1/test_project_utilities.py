import json
import project_utilities as utils
from project_utilities import pd, np
import seaborn as sns
import matplotlib.pyplot as plt
import os

file_name_data = 'proj1_data.json'

with open(file_name_data) as fp:
    data = json.load(fp)

X = utils.process_data(data)
X_train = X[0:10000]
X_validation = X[10000:11000]
X_test = X[11000:]
assert len(X_train) == 10000 , 'Expected 10000. Got %d' % len(X_train)
assert len(X_validation) == 1000 , 'Expected 1000. Got %d' % len(X_validation)
assert len(X_test) == 1000 , 'Expected 1000. Got %d' % len(X_test)

top_words_train = utils.get_top_words(X_train)
top_words_validation = utils.get_top_words(X_validation)
assert len(top_words_train) == 160, 'Expected 160. Got %d' % len(top_words_train)
assert len(top_words_validation) == 160, 'Expected 160. Got %d' % len(top_words_validation)

X_train = utils.insert_top_words_count(X_train, top_words_train)
X_validation = utils.insert_top_words_count(X_validation, top_words_validation)

X_train = pd.DataFrame(X_train)
X_validation = pd.DataFrame(X_validation)

model = utils.LinearRegressionModel()

X_train['bias'] = pd.Series(np.ones(X_train.shape[0]), index=X_train.index)
y_train = X_train['popularity_score']
X_train.drop(columns=['text', 'popularity_score'], inplace=True)

X_train['children_square'] = X_train['children']**2

# Eta (Initial step size) must be significantly smaller when using non-linear features
model.fit(X_train.values, y_train.values, eta=0.000001, epsilon=0.0001)