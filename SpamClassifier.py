import numpy as np
import scipy.io.matlab
import DecisionTree as dt
import RandomForest as rf
import random
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
    classifier = dt.DecisionTree(45)
    classifier.train(X, y)
    y_hat = classifier.predict(test_dat)

    f = open("spam_predictions_decision_tree.csv", 'w')
    f.write("Id,Category\n")
    for i in range(np.size(test_dat, 0)):
        f.write(str(i + 1) + "," + str(int(y_hat[i, 0])) + "\n")
    f.close()
    print("DONE")


def random_forests_classification(X, y, test_dat):
    classifier = rf.RandomForest(10, round(math.sqrt(np.size(X, 1))), np.size(X, 0))
    classifier.train(X, y)
    y_hat = classifier.predict(test_dat)
    f = open("spam_predictions_random_forest.csv", 'w')
    f.write("Id,Category\n")
    for i in range(np.size(test_dat, 0)):
        f.write(str(i + 1) + "," + str(int(y_hat[i, 0])) + "\n")
    f.close()
    print("DONE")


mat = scipy.io.loadmat("spam-dataset/spam_data.mat")
training_data = np.matrix(mat["training_data"])
training_labels = np.matrix(mat["training_labels"]).getT()
test_data = np.matrix(mat["test_data"])

# k_cross_validation(10, np.size(training_data, 0), training_data, training_labels,rf.RandomForest(15,
# int(round(math.sqrt(np.size(training_data, 1)))),int(np.size(training_data, 0))))

# k_cross_validation(10, np.size(training_data, 0), training_data, training_labels, rf.RandomForest(10, round(math.sqrt(np.size(training_data, 1))), np.size(training_data, 0)))

# decision_tree_classification(training_data, training_labels, training_data[1])
random_forests_classification(training_data, training_labels, training_data[1])
