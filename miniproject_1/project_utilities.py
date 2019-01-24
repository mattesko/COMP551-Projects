import copy
from collections import Counter
from operator import itemgetter
import numpy as np
import pandas as pd

def process_data(data_list):
    """
    Convert and return text data to lower case, split it by whitespace, and encode the is_root feature

    Arguments:
    data_list -- list of dictionary type data to perform text lower case conversion, text splitting, and is_root encoding
    
    Return:
    List of dictionary type data that's been processed
    """
    # Deep copy data as to avoid overwriting which could cause unintented side effects
    result = copy.deepcopy(data_list)
    for index, value in enumerate(result):
        result[index]['text'] = value['text'].lower().split(' ')
        result[index]['is_root'] = int(value['is_root'] == 'true')
        
    return result

def concatenate_all_text(data_list):
    """
    Concatenate and return each datapoint's text key value into one list

    Arguments:
    data_list -- list of dictionary type data to concatenate text from

    Return:
    List of all text from each data point from data_list
    """
    all_text = []
    for index, value in enumerate(data_list):
        all_text.extend(value['text'])
    
    return all_text

def get_top_words(data_list, n_top_words=160):
    """
    Get list of top words from given dataset

    Arguments:
    data_list -- Dataset to determine top words from 
    n_top_words -- Number of top words (default 160)

    Return:
    List of strings of the top 160 words
    """
    top_words = []
    
    d = Counter(concatenate_all_text(data_list))
    d_sorted = sorted(d.items(), key=itemgetter(1), reverse=True)
    
    assert len(d_sorted) >= n_top_words, 'Too many top words'
    
    for i in range(n_top_words):
        top_words.append(d_sorted[i][0])
        
    return top_words

def count_top_words(data_point, top_words):
    """
    Count the occurences of words from a given list within a given data point's text

    Arguments:
    data_point -- Data point to count occurences of words from
    top_words -- List of words to count occurences of in data point's text

    Return:
    List of occurences of the words from the data point
    """
    word_count = np.zeros(len(top_words))
    
    for index, word in enumerate(top_words):
        word_count[index] = data_point['text'].count(word)
    
    return word_count

def insert_top_words_count(data_list, top_words):
    """
    Insert columns of top words count to each row within the given dataset

    Arguments:
    data_list -- Dataset to include the top words occurences
    top_words -- List of top occuring words in the dataset

    Return:
    New data set with included columns of occurences of top words
    """
    result = copy.deepcopy(data_list)
    
    for index_example, example in enumerate(result):
        top_words_count = count_top_words(example, top_words)
        
        for index_word, word in enumerate(top_words_count):
            column_name = 'top_word_' + str(index_word + 1).zfill(3)
            result[index_example][column_name] = np.int32(top_words_count[index_word])
    
    return result

def mse(y_true, y_predicted):
    """
    Compute Mean Squared Error of true and predicted values

    Arguments:
    y_true -- The true values
    y_predicted -- The predicted values

    Return:
    Mean Squared Error value
    """
    return ((y_true - y_predicted) ** 2).mean(axis=0)

class LinearRegressionModel:

    weight_estimates = None

    def fit(self, X_values, y_values, beta=10, eta=0.001, epsilon=0.01):
        assert X_values.shape[0] == len(y_values) , 'Number of rows in X must equal to length of y'

        rows, columns = X_values.shape
        weights_old = np.ones(columns)
        weights = np.zeros(columns)
        i = 1

        while True:

            alpha = eta / (1 + beta * i)
            weights = weights_old - 2 * alpha * (np.dot(np.dot(X_values.T, X_values), weights_old) - np.dot(X_values.T, y_values))

            weights_diff_norm = np.linalg.norm(weights - weights_old)
            weights_old = weights
            i+=1
            
            if weights_diff_norm < epsilon:
                break
        
        self.weight_estimates = weights
        return self

    def predict(self, X_values):
        """
        Predict values using linear regression model's estimated weights solution

        Arguments:
        X_values -- Dataset to solve predictions on

        Return:
        List of predictions for each example within X_values
        """
        assert X_values.shape[1] == len(self.weight_estimates), 'Expected dataset of ' + str(len(self.weight_estimates)) + ' number of features but got ' + str(X_values.shape[1])

        return np.dot(X_values, self.weight_estimates)
