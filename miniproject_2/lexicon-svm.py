import os
import json
import numpy as np
from sklearn import svm
from sklearn import metrics
from textblob import TextBlob
from sklearn.model_selection import train_test_split
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer


def load_data():
    filename_neg = "neg.txt"
    filename_pos = "pos.txt"
    filename_neg_words = "negative-words.txt"
    filename_pos_words = "positive-words.txt"

    #Generate sentiment words list
    with open(filename_neg_words) as f:
        content = f.readlines()
    neg_words = [x.strip() for x in content]

    with open(filename_pos_words) as f:
        content = f.readlines()
    pos_words = [x.strip() for x in content]

    all_words = neg_words + pos_words

    # Load pos neg text
    with open(filename_neg) as f:
        content = f.readlines()
    neg_text = [x.strip() for x in content]

    with open(filename_pos) as f:
        content = f.readlines()
    pos_text = [x.strip() for x in content]

    return all_words, neg_text, pos_text

# Feature data generation
def generate_matrix(all_words, neg_text, pos_text):
    total_data_size = len(pos_text) + len(neg_text)
    pos_size = len(pos_text)
    feature_size = len(all_words)
    x_matrix = np.zeros( shape=(total_data_size, feature_size))
    for index, text in enumerate(pos_text):
        if(index % 1000 == 0):
          print("1000")
        text = text.lower()
        text = text.split()
        for word_index, word in enumerate(all_words):
            x_matrix[index][word_index] = text.count(word)
    for index, text in enumerate(neg_text):
        if(index % 1000 == 0):
          print("1000")
        text = text.lower()
        text = text.split()
        for word_index, word in enumerate(all_words):
            x_matrix[pos_size+index][word_index] = text.count(word)
    return x_matrix

def generate_matrix_test(all_words, test_data):
    total_data_size = len(test_data)
    feature_size = len(all_words)
    x_matrix = np.zeros( shape=(total_data_size, feature_size))
    for index, text in enumerate(test_data):
        text = text.lower()
        for word_index, word in enumerate(all_words):
            x_matrix[index][word_index] = text.count(word)
    return x_matrix


def main():
    all_words, neg_text, pos_text= load_data()
    print("Data Loaded")
    x_matrix = generate_matrix(all_words, neg_text, pos_text)
    y_matrix = [1]* len(pos_text) + [0]* len(neg_text)
    print("Training data generated")

    X_train, X_test, y_train, y_test = train_test_split(x_matrix, y_matrix, test_size=0.20,random_state=None)
    start = time.time()
    # tfidf_transformer = TfidfTransformer().fit(X_train)
    # X_train_tfidf = tfidf_transformer.transform(X_train)
    # X_test_tfidf = tfidf_transformer.transform(X_test)

    normalizer_tranformer = Normalizer().fit(X=X_train)
    X_train_normalized = normalizer_tranformer.transform(X_train)
    X_test_normalized = normalizer_tranformer.transform(X_test)

    clf = svm.LinearSVC().fit(X_train_normalized, y_train)
    y_pred = clf.predict(X_test_normalized)
    diff = time.time()-start
    print(metrics.classification_report(y_test, y_pred,digits=5))
    print("time: {}".format(diff))


if __name__=="__main__":
   main()