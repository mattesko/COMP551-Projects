import os
import json
import numpy as np
from sklearn import svm
from sklearn import metrics
from textblob import TextBlob
from sklearn.model_selection import train_test_split

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

    # Load test set
    test_path="../test/test/"
    test_text = []
    for filename in os.listdir(test_path):
        with open(test_path + filename, 'r') as content_file:
            content = content_file.read()
            test_text.append(content)
    return all_words, neg_text, pos_text, test_text

# Feature data generation
def generate_matrix(all_words, neg_text, pos_text):
    total_data_size = len(pos_text) + len(neg_text)
    pos_size = len(pos_text)
    feature_size = len(all_words)
    x_matrix = np.zeros( shape=(total_data_size, feature_size))
    y_matrix = np.zeros( shape=(total_data_size,1))
    for index, text in enumerate(pos_text):
        text = text.lower()
        for word_index, word in enumerate(all_words):
            x_matrix[index][word_index] = text.count(word)
            y_matrix[index][0] = 1
    for index, text in enumerate(neg_text):
        text = text.lower()
        for word_index, word in enumerate(all_words):
            x_matrix[pos_size+index][word_index] = text.count(word)
            y_matrix[pos_size+index][0] = 0
    return x_matrix, y_matrix

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
    all_words, neg_text, pos_text, test_text= load_data()
    print("Data Loaded")
    x_matrix, y_matrix = generate_matrix(all_words, neg_text, pos_text)
    print("Training data generated")
    # kaggle_test_data = generate_matrix_test(all_words, test_text)
    print("Kaggle testing data generated")
    X_train, X_test, y_train, y_test = train_test_split(x_matrix, y_matrix, test_size=0.20,random_state=None)
    print("Kernal creation")
    clf = svm.SVC(kernel='linear')
    print("Train start")
    clf.fit(X_train, y_train)
    print("Train ended")
    y_pred = clf.predict(X_test)
    # kaggle_res = clf.predict(kaggle_test_data)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


if __name__=="__main__":
   main()