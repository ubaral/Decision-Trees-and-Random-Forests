import numpy as np
import math
import random
from collections import defaultdict


class DecisionTree:
    # Node data structure for the Tree Structure
    class Node:
        def __init__(self, left, right, lab, spl_rul):
            self.left_child = left
            self.right_child = right
            self.label = lab
            # Tuple of the split rule (feature index, feature value threshold), None implies this Node is a leaf
            self.split_rule = spl_rul

        def is_leaf(self):
            return (self.left_child is None) and (self.right_child is None)

    # Constructor
    def __init__(self, depth=float("inf")):
        self.root = None
        self.depth_lim = depth

    # Takes in the result of a split, and outputs the weighted average entropy of the given split
    # Could use different impurity measure, but standard entropy seems to work well... i think.
    @staticmethod
    def impurity(left_label_hist, right_label_hist):
        def entropy(label_hist):
            avg_surprise = 0
            set_size = sum([label_hist[key] for key in label_hist])
            for label in label_hist:
                if label_hist[label] > 0:
                    prob = label_hist[label] / set_size
                    surprise = -math.log(prob, 2)
                    avg_surprise += prob * surprise
            return avg_surprise

        left_size = len(left_label_hist)
        right_size = len(right_label_hist)
        weighted_avg_entropy = left_size * entropy(left_label_hist) + right_size * entropy(right_label_hist)
        weighted_avg_entropy /= (left_size + right_size)
        return weighted_avg_entropy

    @staticmethod
    def extract_subsets(data, labels, sample_index):
        subset_data = np.matrix(np.empty((0, np.size(data, 1))))
        subset_labels = np.matrix(np.empty((0, 1)))
        for i in sample_index:
            subset_labels = np.append(subset_labels, labels[i], 0)
            subset_data = np.append(subset_data, data[i], 0)
        return subset_data, subset_labels

    # This function figures out the split rule for a node using the impurity measure and input data
    # Only works for  continuous-valued featured data, we cannot impose such an ordering on categorical variables
    @staticmethod
    def segmenter(data, labels, feature_index_set):
        # We will loop through all the possible splits, and then choose the split that maximizes the information gain,
        # H(curren+t S) - H_after, so minimize the H_after. Since H(current_s) will remain constant during this call
        # Find the impurity for each split and store a tuple of smallest split and impurity.

        curr_min = (float("inf"), None, None)
        for feature in feature_index_set:
            # For a given feature, loop through possible thresholds(i.e values it can take on), which is the domain of
            # this feature
            domain = dict()
            for i in range(np.size(data, 0)):
                feat_val_of_samp = data[i, feature]
                lab_of_samp = labels[i, 0]
                if feat_val_of_samp not in domain.keys():
                    domain[feat_val_of_samp] = defaultdict(int)
                domain[feat_val_of_samp][lab_of_samp] += 1

            # The keys to the domain -> hist map are the possible thresholds, so radix sort, and scan through list
            # in order, and we can update entropy in O(1) time since we will not have to loop
            sorted_feat_vals = list(domain.keys())
            sorted(sorted_feat_vals)
            num_poss_splits = len(sorted_feat_vals) - 1
            if num_poss_splits == 0:
                # There is only one unique value for given feature among all samples, so a split would be pointless,
                # so we will just move onto the next feature instead
                continue

            # Set up the initial left and right histograms, so left is on the first value, and right is on everything to
            # the right of this sorted value, we can update entropy in constant time by scanning through the sorted list
            left_hist = domain[sorted_feat_vals[0]]
            right_hist = domain[sorted_feat_vals[1]]
            if num_poss_splits > 1:
                for k in range(2, len(sorted_feat_vals)):
                    for poss_class in domain[sorted_feat_vals[k]]:
                        right_hist[poss_class] += domain[sorted_feat_vals[k]][poss_class]

            for i in range(1, len(sorted_feat_vals)):
                # Left hist and right hist are preconfigured for the start of loop, so we can check the impurity
                if sum([left_hist[key] for key in left_hist]) == 0 or sum([right_hist[key] for key in left_hist]) == 0:
                    break
                impurity_val = DecisionTree.impurity(left_hist, right_hist)
                if impurity_val < curr_min[0]:
                    curr_min = (impurity_val, feature, sorted_feat_vals[i])
                # Set up the left and right hist for the next run of the loop:
                for class_poss in domain[sorted_feat_vals[i]]:
                    left_hist[class_poss] += domain[sorted_feat_vals[i]][class_poss]  # add to the left
                    right_hist[class_poss] -= domain[sorted_feat_vals[i]][class_poss]  # subtract from the right
        # Return tuple of feature index and threshold that we deemed to have the minimum impurity
        return curr_min[1], curr_min[2]

    # Actually grow the decision tree and add nodes based on the labeled training data
    # m_feature_selection paramater is for attribute bagging when doing random forests
    def train(self, data, labels, m_feature_selection=0):
        self.root = None  # reset the root to null

        def grow_tree(sample_index, d):
            # Different Stopping conditions for better performance
            if d == 0:
                sub = self.extract_subsets(data, labels, sample_index)
                return DecisionTree.Node(None, None, np.argmax(np.histogram(sub[1], bins=[0, .5, 1])[0]), None)
            if len(sample_index) <= 10:
                sub = self.extract_subsets(data, labels, sample_index)
                return DecisionTree.Node(None, None, np.argmax(np.histogram(sub[1], bins=[0, .5, 1])[0]), None)
            if len(sample_index) == 1:
                return DecisionTree.Node(None, None, labels[sample_index[0], 0], None)

            # check if all labels are the same for the given sample_index list
            identical = True
            for i in sample_index:
                if labels[i, 0] != labels[sample_index[0], 0]:
                    identical = False
                    break
            if identical:
                # if all the labels are the same in this (sub)set then we return a leaf node.
                return DecisionTree.Node(None, None, labels[sample_index[0], 0], None)
            else:
                # we need to split samples according to some rule recursively
                data_subset, label_sub = self.extract_subsets(data, labels, sample_index)
                if (np.histogram(label_sub, bins=[0, .5, 1])[0][0] / len(sample_index) > .95) or (np.histogram(label_sub, bins=[0, .5, 1])[1][0] / len(sample_index) > .95):
                    return DecisionTree.Node(None, None, np.argmax(np.histogram(label_sub, bins=[0, .5, 1])[0]), None)
                feature_set = [feat for feat in range(np.size(data_subset, 1))]
                if m_feature_selection > 0:
                    feature_set = random.sample(feature_set, m_feature_selection)
                split = self.segmenter(data_subset, label_sub, feature_set)
                feature_to_split = split[0]
                threshold = split[1]
                if feature_to_split is None or threshold is None:
                    # No good split exists that lessens entropy so we will make this node a leaf and
                    # for the label, we will simply take the class with majority among (sub)samples
                    return DecisionTree.Node(None, None, np.argmax(np.histogram(label_sub, bins=[0, .5, 1])[0]), None)

                left_samps = list()
                right_samps = list()
                for i in sample_index:
                    if data[i, feature_to_split] < threshold:
                        left_samps.append(i)
                    else:
                        right_samps.append(i)

                if len(left_samps) == 0:
                    right_sub = self.extract_subsets(data, labels, right_samps)
                    return DecisionTree.Node(None, None, np.argmax(np.histogram(right_sub[1], bins=[0, .5, 1])[0]),
                                             None)
                if len(right_samps) == 0:
                    left_sub = self.extract_subsets(data, labels, left_samps)
                    return DecisionTree.Node(None, None, np.argmax(np.histogram(left_sub[1], bins=[0, .5, 1])[0]), None)

                return DecisionTree.Node(grow_tree(left_samps, d - 1), grow_tree(right_samps, d - 1), None, split)

        self.root = grow_tree([i for i in range(np.size(data, 0))], self.depth_lim)

    # Traverses the tree rooted at node to find the best label, which is the label at the leaf node we end  up at
    @staticmethod
    def trickle_down(node, sample_vector):
        # spam_index = ['pain', 'private', 'bank', 'money', 'drug', 'spam', 'prescription', 'creative', 'height', 'featured', 'differ',
        #  'width', 'other', 'energy', 'business', 'message', 'volumes', 'revision', 'path', 'meter', 'memo', 'planning',
        #  'pleased', 'record', 'out', ';', '$', '#', '!', '(', '[', '&']
        if node.is_leaf():
            print("data point at leaf, and labeled: {0}".format(node.label))
            return node.label
        else:
            if sample_vector[0, node.split_rule[0]] < node.split_rule[1]:
                print("('{0}') < {1} ".format(node.split_rule[0], node.split_rule[1]))
                return DecisionTree.trickle_down(node.left_child, sample_vector)
            else:
                print("('{0}') >= {1} ".format(node.split_rule[0], node.split_rule[1]))
                return DecisionTree.trickle_down(node.right_child, sample_vector)

    def predict(self, data):
        output = np.matrix(np.empty((0, 1)))
        for sample_row in data:
            output = np.append(output, np.matrix(self.trickle_down(self.root, sample_row)), 0)
        return output
