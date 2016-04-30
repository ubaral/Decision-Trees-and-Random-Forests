import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from DecisionTree import DecisionTree
from RandomForest import RandomForest
from sklearn.feature_extraction import DictVectorizer
import random
import pandas as pd
import math


def k_cross_validation(k, N, X, y, classifier):
    indexSet = set([i for i in range(N)])

    partitionDict = dict()
    for i in range(k):
        partitionDict[i] = [np.matrix(np.empty((0, np.size(X, 1)), X[0, 0].dtype)),
                            np.matrix(np.empty((0, 1), y[0, 0].dtype))]
    for _ in range(N):
        partitionKey = _ % k
        i = random.sample(indexSet, 1)[0]
        indexSet.remove(i)
        partitionDict[partitionKey][0] = np.append(partitionDict[partitionKey][0], X[i], axis=0)
        partitionDict[partitionKey][1] = np.append(partitionDict[partitionKey][1], y[i], axis=0)

    for key in partitionDict:
        validationPartition = partitionDict[key]
        validation_X = validationPartition[0]
        validation_y = validationPartition[1]

        train_X = np.matrix(np.empty((0, np.size(X, 1)), X[0, 0].dtype))
        train_y = np.matrix(np.empty((0, 1), y[0, 0].dtype))

        for otherKey in partitionDict:
            if otherKey != key:
                trainingPartition = partitionDict[otherKey]
                train_X = np.append(train_X, trainingPartition[0], axis=0)
                train_y = np.append(train_y, trainingPartition[1], axis=0)

        classifier.train(train_X, train_y)
        y_hat = np.matrix(classifier.predict(validation_X), dtype=validation_y.dtype)
        print("validation error rate is: {0}".format(
            (np.sum(np.bitwise_xor(y_hat, validation_y)) / np.size(validation_y, 0))))


def decision_tree_classification(X, y, test_dat):
    classifier = DecisionTree(45)
    classifier.train(X, y)
    y_hat = classifier.predict(test_dat)

    f = open("census_predictions_decision_tree.csv", 'w')
    f.write("Id,Category\n")
    for i in range(np.size(test_dat, 0)):
        f.write(str(i + 1) + "," + str(int(y_hat[i, 0])) + "\n")
    f.close()
    print("DONE")


def random_forests_classification(X, y, test_dat):
    classifier = RandomForest(20, round(math.sqrt(np.size(X, 1))), np.size(X, 0))
    # classifier = RandomForest(1, round(math.sqrt(np.size(X, 1))), 100, 45)
    classifier.train(X, y)
    y_hat = classifier.predict(test_dat)
    f = open("census_predictions_random_forest.csv", 'w')
    f.write("Id,Category\n")
    for i in range(np.size(test_dat, 0)):
        f.write(str(i + 1) + "," + str(int(y_hat[i, 0])) + "\n")
    f.close()
    print("DONE")


def parseCSV(df):
    data = np.matrix(np.empty((df.last_valid_index() + 1, 0)))
    keylist = []
    labels = []
    for key in df:
        keylist.append(key)
        if key == "label":
            labels = np.matrix(df[key]).getT()

        elif "?" in np.array(df[key]):
            le = LabelEncoder()
            le.fit(np.array(df[key]))
            imp = Imputer(missing_values=le.transform("?"), strategy='most_frequent', axis=1)
            tform = le.transform(np.array(df[key]))
            imp.fit_transform(tform)
            toapp = le.inverse_transform(np.array(imp.fit_transform(tform), dtype=int))
            data = np.append(data, np.matrix(toapp).getT(), axis=1)
        else:
            data = np.append(data, np.matrix(df[key]).getT(), axis=1)

    return data, labels, keylist


panda_df = pd.read_csv('census_data/train_data.csv')
dataset, train_labels, keylistt = parseCSV(panda_df)
keylistt.remove('label')

panda_df2 = pd.read_csv('census_data/test_data.csv')
test_dataset, test_labels, keylist_test = parseCSV(panda_df2)

dict_row_list = []
for row in dataset:
    dict_row_list.append(dict(zip(keylistt, row.tolist()[0])))
v = DictVectorizer(sparse=False)
train_data = np.matrix(v.fit_transform(dict_row_list))

dict_row_list = []
for row in test_dataset:
    dict_row_list.append(dict(zip(keylist_test, row.tolist()[0])))
test_data = np.matrix(v.transform(dict_row_list))

# N = np.size(train_data[0:1000], 0)
# decision_tree_classification(train_data[0:4000], train_labels[0:4000], test_data)
random_forests_classification(train_data[0:3000], train_labels[0:3000], test_data)
#decision_tree_classification(train_data[0:1000], train_labels[0:1000], test_data)
