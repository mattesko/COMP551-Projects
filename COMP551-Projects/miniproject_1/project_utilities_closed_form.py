import copy
from collections import Counter
from operator import itemgetter
import numpy as np
import pandas as pd
import nltk

#nltk.download('vader_lexicon')  # if you want to use  nltk.sentiment.vader, you need excute it 

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
        #result[index]['sentiment'] = abs(nltk_sentiment(value['text'])['compound'])  # run it if you want to add new feature "sentiment"
        result[index]['text'] = value['text'].lower().split(' ')
        result[index]['is_root'] = int(value['is_root'] == True)
        
    return result

def nltk_sentiment(sentence):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    nltk_sentiment = SentimentIntensityAnalyzer()
    score = nltk_sentiment.polarity_scores(sentence)

    return score

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

def get_top_words(data_list, n_top_words):
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

def add_new_children_feature (data_list, power):
    
    """
    Function: Add a new feature column about the nth power of "children" 's value
    
    Argument:
    data_list -- Dataset to include the "children" value
    power -- number of power of the "children"
    
    Return:
    A new data set include a new column about nth power of "children" 's value
    
    """
    result = copy.deepcopy(data_list)
    new_children_feature = [0]*len(data_list)
    for index, value in enumerate (result):
        new_children_feature[index] = (value['children'])** power
        result[index]['new_children_feature'] = new_children_feature[index]
    return result

    
def add_marks_feature (data_list):
    
    result = copy.deepcopy(data_list)
    for index, value in enumerate (result):
        result[index]['has_exclamation'] = int("!" in value['text']) 
        result[index]['has_question_mark'] = int("?" in value['text'])
        #result[index]['has_period'] = int("." in value['text'])
    return result

def closedformLinearRegression (X,Y):
    """
    Function: Used to calculated the prediction output by linear Regression closed form
    
    Argument:
    X -- It is input of the training set, used to calculate weighting coeffcient
    Y -- It is output of the training set, used to calculate weighting coeffcient
    x -- It is the testing data set
    
    Return:
    prediction -- the predicted output of the testing data x
    """
    
    XT = X.transpose()
    XTX = XT.dot(X)
    XTX_inverse = np.linalg.inv(XTX)
    XTY = XT.dot(Y)
    weight_coefficient = np.matmul(XTX_inverse,XTY)
    
    return weight_coefficient
def making_prediction (X,weight_coefficient):

    pridiction = np.matmul(X,weight_coefficient)
    return pridiction

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

