import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def get_indexes_from_names(df, names):
    indexes = []
    for item in names:
        indexes.append(df.columns.get_loc(item))
    return indexes

"""--------------------------------------------------------------------------------------------------------------------------------------"""

def process_dataframe(df):
    new_df = pd.DataFrame()
    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            if df[col].nunique() == 2:
                unique_values = df[col].unique()
                tmp = df[col].map({unique_values[0]:0, unique_values[1]:1})
            elif df[col].nunique() > 2 and df[col].nunique() <= 5:
                tmp = pd.get_dummies(df[col], prefix=col)
            elif df[col].nunique() > 5:
                continue
            new_df = pd.concat([new_df, tmp], axis=1)
        else:
            if df[col].nunique() > 5:
                continue
            elif df[col].nunique() > 2 and df[col].nunique() <= 5:
                tmp = pd.get_dummies(df[col], prefix=col)
                new_df = pd.concat([new_df, tmp], axis=1)
            else:
                new_df = pd.concat([new_df, df[col]], axis=1)
    return new_df

def format_dataset(x_df, y_df, sensitive_groups, sensitive_labels):
    
    """
    Format pandas dataframe to clean it, and return numpy arrays
    
    @dataset : the dataset to format (pandas.df)
    @labels_name : The labels name (list)
    @sensitive_groups : the sensitive groups name (list)
    @sensitive_labels : the sensitive labels name (list)
    returns df_labels(labels dataframe), df_features(labels dataframe), features_name (list), x (features matrix), y (labels matrix), sgroup_indexes(list of indexes), slabels_indexes(list of indexes)
    """
    X_ = x_df.copy()
    X_.insert(0, "starts with", np.ones(len(X_.index)))
    X_ = process_dataframe(X_)
    X_ = X_.astype(int)
    y_df = pd.get_dummies(y_df)
    y = y_df.to_numpy()                                                                    # Get the labels as a numpy array
    x = X_.to_numpy()                                                                  # Get the features as a numpy array                                        
    sgroup_indexes = get_indexes_from_names(X_, sensitive_groups)                      # Get the indexes of the sensitives groups (index from feature list)
    slabels_indexes = get_indexes_from_names(y_df, sensitive_labels)                       # Get the indexes of the sensitives labels (index from label list)
    return x, y, sgroup_indexes, slabels_indexes

def format_dataset_(df, labels_name, sensitive_groups, sensitive_labels):
    
    """
    Format pandas dataframe to clean it, and return numpy arrays
    
    @dataset : the dataset to format (pandas.df)
    @labels_name : The labels name (list)
    @sensitive_groups : the sensitive groups name (list)
    @sensitive_labels : the sensitive labels name (list)
    returns df_labels(labels dataframe), df_features(labels dataframe), features_name (list), x (features matrix), y (labels matrix), sgroup_indexes(list of indexes), slabels_indexes(list of indexes)
    """
    
    df.dropna()                                                                                # SLIM doestn't handle NA
    df.drop_duplicates()                                                                       # Remove duplicates lines for data reduction  
    df.insert(0, "starts with", np.ones(len(df.index)))
    
    features_name = [item for item in df.columns.to_numpy() if item not in labels_name]        # Get the features names

    df_labels = df[labels_name]
    df_features = df[features_name]
        
    y = df_labels.to_numpy()                                                                    # Get the labels as a numpy array
    x = df_features.to_numpy()                                                                  # Get the features as a numpy array                                        
    sgroup_indexes = get_indexes_from_names(df_features, sensitive_groups)                      # Get the indexes of the sensitives groups (index from feature list)
    slabels_indexes = get_indexes_from_names(df_labels, sensitive_labels)                       # Get the indexes of the sensitives labels (index from label list)
    return df_labels, df_features, features_name, x, y, sgroup_indexes, slabels_indexes
    
"""--------------------------------------------------------------------------------------------------------------------------------------"""

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
    most_common_item = max(y, key = y.tolist().count)
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

"""--------------------------------------------------------------------------------------------------------------------------------------"""

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

"""--------------------------------------------------------------------------------------------------------------------------------------"""

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

"""--------------------------------------------------------------------------------------------------------------------------------------"""

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

"""--------------------------------------------------------------------------------------------------------------------------------------"""

def print_scoring_system_multiclass(labels_name, features_name, l_lists):

    """
    Print the scoring systems according to the features names and the lambda coefficients
    
    @label_names are the name of the labels to print for each the scoring system
    @features_names are the name of the features of the dataset
    @l is the matrix of the coefficients lambda for the scoring systems
    
    is l[i] == 0 then the row is not printed 
    """
    
    for index, l in enumerate(l_lists):
        print("\n")
        print(f"SCORE FOR {labels_name[index]} \n")
        print("*" * 85)
        for i in range(0, len(l)):    
            if(l[i]):
                print("* " + features_name[i] + " ?" + " " * (70 - len(features_name[i]) - len(str(l[i]))) + str(l[i]) + " POINTS   *")
                print("*" * 85)
        print("\n")

"""---------------------------------------------------------------------------------------------------------------------------------------------"""

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

"""--------------------------------------------------------------------------------------------------------------------------------------"""

def get_balanced_accuracy(x, y, l_lists):

    y_pred = predict(x, l_lists)
    y = format_labels(y)
    y_pred = format_labels(y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
                
    return balanced_accuracy

"""--------------------------------------------------------------------------------------------------------------------------------------"""