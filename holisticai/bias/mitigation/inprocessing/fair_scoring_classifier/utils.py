import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def get_indexes_from_names(df, names):
    indexes = []
    for item in names:
        indexes.append(df.columns.get_loc(item))
    return indexes


def process_y(y):
    """
    Converts from one-hot encoding to a list of labels
    
    @y : the labels matrix (np.array)
    returns y_processed(list of labels)
    """
    y_processed = []
    for line in y:
        for i in range(len(line)):
            if line[i] == 1:
                y_processed.append(i)
    return y_processed


def get_majority_class(y):
    """
    Returns the majority class and its accuracy
    params:
        y : the list of labels (list)
        returns : most_common_item (int), acc (float)
    """
    y = np.array(y).flatten()
    most_common_item = max(y, key=y.tolist().count)
    acc = np.count_nonzero(y == most_common_item) / len(y)
    return most_common_item, acc


def get_initial_solution(y):
    """
    Calculates the accuracy of the majority class

    @y : the labels matrix (np.array)
    returns (most_common_item (int), acc (float)) (tuple)
    """
    y_processed = process_y(y)
    return get_majority_class(y_processed)


def remove_inconcsistency(x, y):
    
    """
    Remove the inconsistencies

    @x : The dataset features (np.array)
    @y : The dataset labels (np.array)
    
    return the dataset withtout the inconsistencies
    """

    x = x.tolist()
    y = y.tolist()

    new_x = []
    new_y = []

    for i in range(len(x)):
        target = get_max_y(x[i], x, y)
        if y[i][target] == 1:
            new_x.append(x[i])
            new_y.append(y[i])

    return np.array(new_x), np.array(new_y)


def get_max_y(cur_x, x, y):
    y = [y[i] for i, x in enumerate(x) if x == cur_x]

    counts = [0 for i in range(len(y[0]))]

    for i in range(len(y)):
        y[i] = y[i].index(1)

    for i in range(len(y)):
        counts[y[i]] += 1

    return counts.index(max(counts))


def get_class_count(y):

    """
    Get the count for each class in the labels

    return the count of each class
    """

    count = []

    for i in range(len(y[0])):
        count.append(0)

    for labels in y:
        for index, label in enumerate(labels):
            if label == 1:
                count[index] += 1

    return count


def get_class_indexes(y):

    """
    Get the indexes of each class in the labels
    params:
    y : the labels matrix (np.array)
    return the indexes of each class (list of list)
    """

    indexes = []

    for i in range(len(y[0])):
        indexes.append([])

    for i, labels in enumerate(y):
        for index, label in enumerate(labels):
            if label == 1:
                indexes[index].append(i)

    return indexes


def predict(x, l_lists):

    """
    Returns the predictions of the set of scoring systems for the given entries

    @x : The input features matrix
    @l_lists : The set of scoring systems
    """

    y = []

    for index, sample in enumerate(x):
        scores = []
        for l_list in l_lists:
            scores.append(sum(feature * l_list[j] for j, feature in enumerate(sample)))
        y_pred = []
        for i in range(len(scores)):
            if i == np.argmax(scores):
                y_pred.append(1)
            else:
                y_pred.append(0)
        y.append(y_pred)

    return np.array(y)


def format_labels(y):
    
    y_formatted = []

    for labels in y:
        for i in range(len(labels)):
            if labels[i] == 1:
                y_formatted.append(i)

    return y_formatted

def get_accuracy(x, y, l_lists):  
    """
    Compute accuracy for the scoring systems : 
    
    @x_test the test set
    @y_test labels of the test set
    @l the matrix of coefficients lambda that represents the scoring systems
    
    return accuracy
    """
    
    y_pred = predict(x, l_lists)
    y = format_labels(y)
    y_pred = format_labels(y_pred)
    accuracy = accuracy_score(y, y_pred)
                
    return accuracy


def get_balanced_accuracy(x, y, l_lists):

    y_pred = predict(x, l_lists)
    y = format_labels(y)
    y_pred = format_labels(y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
                
    return balanced_accuracy
