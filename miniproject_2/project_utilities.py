import os
import re
import numpy as np
import matplotlib.pyplot as plt
import itertools

"""
Contains helper functions for importing train and test data
The two classes are:
    pos -- Encoded as 1
    neg -- Encoded as 0
"""

def import_train_data(path='train'):
    """
    Returns labelled data under the given path

    Arguments:
    path (str) -- Path under which to import data from. (default: 'train')

    Return:
    List of dictionary elements. Elements have 'text' and 'category' fields. 'text' is the review content. 'category' is the label (1 or 0) for whether it's a positive or negative review.
    """
    path_pos = os.path.join(path, 'pos')
    path_neg = os.path.join(path, 'neg')
    train_data = []

    for sub_path in [path_pos, path_neg]:
        category = 1 if sub_path == path_pos else 0

        for file_name in os.listdir(sub_path):
            
            with open(os.path.join(sub_path, file_name), encoding='utf8') as fd:

                train_data.append(
                    {
                        'category' : category,
                        'text' : fd.read()
                    })

    return train_data

def import_test_data(path='test'):
    """
    Returns labelled data under the given path

    Arguments:
    path (str) -- Path under which to import data from. (default: 'test')

    Return:
    List of dictionary elements. Elements have 'text' and 'id' fields. 'text' is the review content. 'id' is a unique identifier for that review, it's needed for properly submitting to the COMP551 MiniProject 2 competition.
    """
    test_data = []

    for file_name in os.listdir(path):
        
        identifier = re.match(r'\d+', file_name).group(0)
        with open(os.path.join(path, file_name), encoding='utf8') as fd:

            test_data.append(
                {
                    'id' : identifier,
                    'text' : fd.read()
                })

    return test_data

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figsize=(10,10)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()