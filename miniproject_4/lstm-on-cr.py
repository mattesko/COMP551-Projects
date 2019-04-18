# from comet_ml import Experiment
import re
import numpy as np
import pandas as pd
import os
import time
import datetime
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

nltk.download('punkt')

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
NUMBER_DIFFERENT_OUTPUTS = 2

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_input_text(text):
    english_stopwords = set(stopwords.words("english"))
    wordnet_lemmatizer = WordNetLemmatizer()
    clean_text = []
    for sent in text:
        clean_sent = ""
        sent_tokens = word_tokenize(sent)
        for token in sent_tokens:
            clean_sent += wordnet_lemmatizer.lemmatize(token) + " " if token not in english_stopwords else ""
        clean_text.append(clean_sent)
    return clean_text

def load_reviews_dataset():
    #src = list(files.upload().values())[0]
    #open('customer review data','wb').write(src)

    products = ["Apex AD2600 Progressive-scan DVD player.txt"
    ,"Canon G3.txt"
    ,"Creative Labs Nomad Jukebox Zen Xtra 40GB.txt"
    ,"Nikon coolpix 4300.txt"
    ,"Nokia 6610.txt"]
    examples = []
    for product in products:
        examples += list(open('./data/customer_reviews/' + product, "r", encoding="utf-8").readlines())
    
    # for every examples, keep the one starting with a ranking
    x_text, y = [],[]
    for example in examples:
        final_label = 0
        temp_split = example.split("##")
        # don't consider unlabeled sentences
        if len(temp_split) <= 1:
            continue
        temp_label, temp_sentence = temp_split
        # parse the temp_label to find positive or negative
        positive_label = temp_label.split("+")
        #print("len positive label: {}".format(len(positive_label)))
        if len(positive_label) > 1:
            final_label = 1
        
        # so the final_label is either 0 or 1, 0 if negative, 1 if positive
        final_sentence = clean_str(temp_sentence.strip())
        x_text.append(final_sentence)
        y.append(final_label)
    return x_text, y
        
data, labels = load_reviews_dataset()
data = clean_input_text(data)

# load pretrain glove word2vec instance for preprocessing
filename = './data/glove.6B.300d.txt'
print('Indexing Glove 6B 300D word vectors.')
embeddings_index = {}
with open(filename, encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))

# vectorize the input text (both negative and positive )
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(labels)

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=VALIDATION_SPLIT)

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM

lstm_out = 200
batch_size = 64

model = Sequential()
model.add(Embedding(num_words,EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix),
                    input_length=MAX_SEQUENCE_LENGTH, trainable=False))
model.add(LSTM(units=lstm_out, activation='relu', 
                dropout=0.5, recurrent_dropout=0.2))
# model.add(LSTM(units=lstm_out, activation='relu', 
#                 dropout=0.5, recurrent_dropout=0.2))
model.add(Dense(NUMBER_DIFFERENT_OUTPUTS, activation='softmax'))
model.compile(
    loss = 'categorical_crossentropy', 
    optimizer='adam',
    metrics = ['accuracy'])
print(model.summary())

# experiment = Experiment(api_key="PqrK4iPuQntpHwzb6SvJuXbdh", project_name="COMP 551", workspace="mattesko")
# experiment.add_tag('LSTM-CR')
# experiment.log_dataset_info('CR')

model.fit(X_train, y_train, batch_size=batch_size, epochs=10,  verbose=10, validation_data=(X_test, y_test))
score, accuracy = model.evaluate(X_test, y_test, verbose=2, batch_size = batch_size)

# experiment.end()
print('Score: %.3f' % score)
print('Validation Accuracy: %.3f' % accuracy)
