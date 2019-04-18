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

def get_data_sst2(filepath):
    data = []
    labels = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            words = line.split(' ')
            labels.append(int(words[0]))
            del words[0]
            data.append(" ".join(words))
    return data, labels
    
train_data, train_labels = get_data_sst2('./data/stanfordSentimentTreebank/stsa.binary.train')
test_data, test_labels = get_data_sst2('./data/stanfordSentimentTreebank/stsa.binary.test')
data = train_data + test_data
labels = train_labels + test_labels

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
# experiment.add_tag('LSTM-SST2')
# experiment.log_dataset_info('SST2')

model.fit(X_train, y_train, batch_size=batch_size, epochs=10,  verbose=10, validation_data=(X_test, y_test))
score, accuracy = model.evaluate(X_test, y_test, verbose=2, batch_size = batch_size)

# experiment.end()
print('Score: %.3f' % score)
print('Validation Accuracy: %.3f' % accuracy)