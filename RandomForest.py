import DecisionTree as dt
import numpy as np
import random
from collections import defaultdict


class RandomForest:
    def __init__(self, num_trees, m, n_prime, depth=float("inf")):
        self.decision_forest = list()
        self.num_trees = num_trees
        self.m = m
        self.n_prime = n_prime
        self.depth = depth
        self.split_counts = defaultdict(int)

    def train(self, data, labels):
        self.decision_forest = list()
        index_set = [k for k in range(np.size(data, 0))]
        for i in range(self.num_trees):
            bagged_sample_list = list()
            # create another tree to put in our forest
            for k in range(self.n_prime):
                # Loop n_prime times and generate another sample_index
                bagged_sample_list.append(random.choice(index_set))
            dt_instance = dt.DecisionTree(self.depth)
            data_sub, labels_sub = dt_instance.extract_subsets(data, labels, bagged_sample_list)
            dt_instance.train(data_sub, labels_sub, self.m)
            self.split_counts[dt_instance.root.split_rule] += 1
            self.decision_forest.append(dt_instance)
        print("splitcounts are {0}".format(self.split_counts))

    def predict(self, data):
        output = np.matrix(np.empty((np.size(data, 0), len(self.decision_forest))))
        for j in range(len(self.decision_forest)):
            output[:, j] = np.matrix(self.decision_forest[j].predict(data))[:, 0]

        y_hat = np.matrix(np.empty((np.size(data, 0), 1)))
        for i in range(np.size(output, 0)):
            y_hat[i, 0] = np.argmax(np.histogram(output[i, :], bins=[0, .5, 1])[0])
        return y_hat
