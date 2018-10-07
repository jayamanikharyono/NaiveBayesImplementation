# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

dataset = pd.DataFrame()
model = {}

def load_dataset(filename):
    #Load Dataset
    global dataset
    dataset = pd.read_csv(filename)
    
def remove_unused_feature(list_features):
    global dataset
    for feature in list_features:
        del dataset[feature]

def replace_nan():
    global dataset
    dataset = dataset.fillna(-999)
        

def build_model(features, labels):
    columns_details = {}
    for column in features.columns:
        columns_details[column] = features[column].value_counts()
    del column
    
    
    #Summarize Class
    print("Class : {0}".format(labels.unique()))
    class_prior = dict(zip(labels.value_counts().index, labels.value_counts().values))
    data_points = sum(class_prior.values())
    class_dict = {}
    for class_ in class_prior.keys():
        class_dict[class_] = class_prior[class_]
        class_prior[class_] = class_prior[class_]/ data_points
    
    #Summarize Features
    feature_columns = list(columns_details.keys())
    feature_prior = {}
    for feature in feature_columns:
        values = columns_details[feature]
        values = values.astype('float')
        for value in values.keys():
            values[value] = values[value]/data_points
        feature_prior[feature] = dict(values)
    
    dataframe_temporary = pd.concat([features, labels], axis = 1)
    for column in feature_columns:
        column_details = pd.DataFrame(columns_details[column])
        for class_ in class_prior.keys():
            number_of_value_feature = []
            for value in feature_prior[column]:
                index_of_class = dataframe_temporary[dataframe_temporary[labels.name] == class_]
                count_of_class = index_of_class[dataset[column] == value].shape[0]
                number_of_value_feature.append(count_of_class)
            column_details[class_] = value = number_of_value_feature
        columns_details[column] = column_details
        
    #calculate probability
    denominator = list(class_dict.values())
    for column in feature_columns:
        feature = columns_details[column]
        feature_dicts = {}
        for index, row in feature.iterrows():
            value = row[1:]
            if 0 in list(row):
                value = [((value[i]+1)/denominator[i]) for i in range(len(value))]
            else:
                value = [(value[i]/denominator[i]) for i in range(len(value))]
            classes = list(class_prior.keys())
            value_dicts = dict(zip(classes, value))
            feature_dicts[index] = value_dicts
        model[column] = feature_dicts
    model[labels.name] = class_prior
    return model
            
def get_predictions(features):
    classes = list(model[Y_test.name].keys())
    class_predictions = []
    for i, row in features.iterrows():
        class_prob = []
        for class_ in classes:
            probabilities = 1
            for index, value in row.iteritems():
                try:
                    probabilities = probabilities * model[index][value][class_]
                except:
                    probabilities = probabilities
            probabilities = probabilities * model[Y_test.name][class_]
            class_prob.append(probabilities)
        index_max = np.argmax(class_prob)
        class_predictions.append(classes[index_max])
    return class_predictions


    
load_dataset("balance-scale.csv")
replace_nan()
Y = dataset['ClassName']
X = dataset.iloc[:,1:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state = 42)
model = build_model(X_train, Y_train)
Y_pred= get_predictions(X_test) 
accuracy = accuracy_score(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)
cm = confusion_matrix(Y_test, Y_pred)
