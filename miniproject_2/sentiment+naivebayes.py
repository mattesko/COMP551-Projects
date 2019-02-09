import pandas as pd
import os
import json
import numpy as np
from textblob import TextBlob


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



# Train the weight matrix
def train (all_words, neg_text, pos_text):
    # Pos matrix
    pos_matrix = np.zeros(shape=( len(pos_text), len(all_words)))
    for index, text in enumerate(pos_text):
        text = text.lower()
        for word_index, word in enumerate(all_words):
            if (word in text):
                pos_matrix[index][word_index] = 1
    pos_matrix_count = np.sum(pos_matrix, axis=0)
    list_prob_pos = []
    for count in pos_matrix_count:
        list_prob_pos.append( (count + 1)/ (len(pos_text) +2.0) )

    #Neg matrix
    neg_matrix = np.zeros(shape=( len(neg_text), len(all_words)))
    for index, text in enumerate(neg_text):
        text = text.lower()
        for word_index, word in enumerate(all_words):
            if (word in text):
                neg_matrix[index][word_index] = 1
    neg_matrix_count = np.sum(neg_matrix, axis=0)
    list_prob_neg = []
    for count in neg_matrix_count:
        list_prob_neg.append( (count + 1)/ (len(neg_text) + 2.0) )
    return list_prob_pos, list_prob_neg


def test(text, pos_prob_list, neg_prob_list, word_list):
    text = text.lower()
    prob_pos = 1
    prob_neg = 1
    for word_index, word in enumerate(word_list):
        if (word in text):
            prob_pos *= pos_prob_list[word_index]
            prob_neg *= neg_prob_list[word_index]
    res = 1 if (prob_pos > prob_neg) else 0
    return res



def main():
    all_words, neg_text, pos_text = load_data()
    list_prob_pos, list_prob_neg = train(all_words, neg_text, pos_text)
    # np.save("prob_pos_list.npy", list_prob_pos)
    # np.save("prob_neg_list.npy", list_prob_neg)
    # list_prob_pos = np.load("prob_neg_list.npy")
    # list_prob_neg = np.load("prob_neg_list.npy")


    # with open('csvfile.csv','w') as file:

    #     test_path="../test/test/"
    #     for filename in os.listdir(test_path):
    #         with open(test_path + filename, 'r') as content_file:
    #             content = content_file.read()
    #             res = test(content, list_prob_pos, list_prob_neg, all_words)
    #             filename_without_ext = filename.split(".")[0]
    #             file.write(filename_without_ext+","+ str(res))
    #             file.write("\n")



if __name__=="__main__":
   main()