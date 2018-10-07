# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
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




def get_mean(values):
    sum = 0
    count = 0
    for value in values:
        sum = sum + value
        count = count + 1
    return sum/count

def get_std(values):
    return np.std(values)

"""
def get_std(values):
    sums = 0
    sum_power = 0
    count = 0
    for value in values:
        sums = sums + value
        sum_power = sum_power + np.power(value,2)
        count = count +1
    return np.sqrt((count * sums - sum_power)/ (count * (count-1)))
"""
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent



def build_model(features, labels):
    
    #Summarize Class
    print("Class : {0}".format(labels.unique()))
    class_prior = dict(zip(labels.value_counts().index, labels.value_counts().values))
    data_points = sum(class_prior.values())
    class_dict = {}
    for class_ in class_prior.keys():
        class_dict[class_] = class_prior[class_]
        class_prior[class_] = class_prior[class_]/ data_points
    model[labels.name] = class_prior

    for column in features.columns:
        class_details = {}
        for class_ in class_prior.keys():
            class_detail =  features[column][labels.values == class_]
            mean = get_mean(class_detail.values)
            std = get_std(class_detail.values)
            class_details[class_] = {'mean':mean, 'std':std}
        model[column] = class_details
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
                    mean, std = model[index][class_]['mean'], model[index][class_]['std']
                    probabilities = probabilities * calculateProbability(value, mean, std)
                except:
                    probabilities = probabilities
            probabilities = probabilities * model[Y_test.name][class_]
            class_prob.append(probabilities)
        index_max = np.argmax(class_prob)
        class_predictions.append(classes[index_max])
    return class_predictions



load_dataset("haberman.csv")
replace_nan()
Y = dataset['SurvivalStatus']
X = dataset.iloc[:,:3]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state = 42)
model = build_model(X_train, Y_train)
Y_pred= get_predictions(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)
cm = confusion_matrix(Y_test, Y_pred)
