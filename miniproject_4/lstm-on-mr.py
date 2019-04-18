from comet_ml import Experiment
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
nltk.download('punkt')

from sklearn.model_selection import train_test_split

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

# load MR (movie reviews with 1 sentence per input)
def load_MR_data_and_labels():
    positive_data_file = "./data/rt-polaritydata/rt-polarity.pos"
    negative_data_file = "./data/rt-polaritydata/rt-polarity.neg"  

    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    #x_text = [clean_str(sent) for sent in x_text]
    # remove stopwords and lemmatize
    x_text = [clean_str(sent) for sent in x_text]
    clean_text = clean_input_text(x_text)
    print("Length of Train Text: %d" %len(x_text))
    print("Length of Clean Text: %d" % len(clean_text))
    # Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [clean_text, y]

x_text, y = load_MR_data_and_labels()
x_text[-1]

# hyperparameters for the cnn dealing with the movie dataset
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
NUMBER_DIFFERENT_OUTPUTS =2
                        
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

print('Vectorizing input text')
# vectorize the input text (both negative and positive )
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(x_text)
sequences = tokenizer.texts_to_sequences(x_text)
word_index = tokenizer.word_index
print(len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(y))

# split the data into a training set and a validation set
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=VALIDATION_SPLIT)

print('Preparing embedding matrix')
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
        
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
                            
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM

embed_dim = 128
lstm_out = 200
batch_size = 64

model = Sequential()
model.add(Embedding(num_words,EMBEDDING_DIM,embeddings_initializer=Constant(embedding_matrix),
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

# For logging Experiment
experiment = Experiment(api_key="PqrK4iPuQntpHwzb6SvJuXbdh", project_name="COMP 551", workspace="mattesko")
experiment.add_tag('LSTM-MR')
experiment.log_dataset_info(name='MR')

model.fit(X_train, y_train, batch_size=batch_size, epochs=10,  verbose=5, validation_data=(X_test, y_test))
score, accuracy = model.evaluate(X_test, y_test, verbose=2, batch_size = batch_size)

experiment.end()
print('Score: %.3f' % score)
print('Validation Accuracy: %.3f' % accuracy)
