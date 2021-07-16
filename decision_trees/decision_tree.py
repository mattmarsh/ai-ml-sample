# CS 6601
# Assignment 4
# Matt Marsh

from math import log
import numpy as np
import random
from copy import deepcopy

class DecisionNode():
    #Class to represent a single node in a decision tree.

    def __init__(self, left, right, decision_function,class_label=None):
        #Create a node with a left child, right child,decision function and optional class label for leaf nodes
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label


    #Return on a label if node is leaf, or pass the decision down to the node's left/right child
    def decide(self, feature):
        if self.class_label is not None:
            return self.class_label

        return self.left.decide(feature) if self.decision_function(feature) else self.right.decide(feature)


def build_decision_tree():
    r = DecisionNode(None, None, lambda feature: feature[0])
    n1 = DecisionNode(None, None, None, 1)
    n2 = DecisionNode(None, None, lambda feature: feature[2])
    n3 = DecisionNode(None, None, lambda feature: feature[3])
    n4 = DecisionNode(None, None, lambda feature: feature[3])
    n5 = DecisionNode(None, None, None, 1)
    n6 = DecisionNode(None, None, None, 0)
    n7 = DecisionNode(None, None, None, 0)
    n8 = DecisionNode(None, None, None, 1)

    r.left = n1
    r.right = n2
    n2.left = n3
    n2.right = n4
    n3.left = n5
    n3.right = n6
    n4.left = n7
    n4.right = n8
    return r


def confusion_matrix(classifier_output, true_labels):
    #output should be [[true_positive, false_negative], [false_positive, true_negative]]
    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0
    for i in range(len(true_labels)):
        if classifier_output[i]:
            if true_labels[i]:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if true_labels[i]:
                false_negative += 1
            else:
                true_negative += 1

    return [[true_positive, false_negative], [false_positive, true_negative]]


def precision(classifier_output, true_labels):
    # precision is measured as: true_positive/ (true_positive + false_positive)
    cm = confusion_matrix(classifier_output, true_labels)
    if cm[0][0] + cm[1][0] == 0:
        return float("NaN")
    return float(cm[0][0]) / (cm[0][0] + cm[1][0])

def recall(classifier_output, true_labels):
    #recall is measured as: true_positive/ (true_positive + false_negative)
    cm = confusion_matrix(classifier_output, true_labels)
    if cm[0][0] + cm[0][1] == 0:
        return float("NaN")
    return float(cm[0][0]) / (cm[0][0] + cm[0][1])

def accuracy(classifier_output, true_labels):
    #accuracy is measured as:  correct_classifications / total_number_examples
    correct = 0
    total = 0
    for i in range(len(true_labels)):
        if classifier_output[i] == true_labels[i]:
            correct += 1
        total += 1
    return float(correct) / total


def entropy(class_vector):
    # Compute the Shannon entropy for a vector of classes
    # Note: Classes will be given as either a 0 or a 1.
    assert len(class_vector) > 0
    q = float(sum(class_vector)) / len(class_vector)
    if q == 0 or q == 1:
        return 0
    else:
        return -(q * log(q, 2) + (1-q) * log(1-q, 2))


def information_gain(previous_classes, current_classes ):
    return entropy(previous_classes) - entropy(current_classes)


class DecisionTree:

    def __init__(self, depth_limit=float('inf'), threshold_method="mean", features_used="all"):
        self.root = None
        self.depth_limit = depth_limit
        self.threshold_method = threshold_method
        self.features_used = features_used

    def fit(self, features, classes):
        self.root = self.__build_tree__(features, classes, self.depth_limit)

    def __find_cut_threshold__(self, feature, classes=None):
        # find cut threshold for continuous valued variable. feature is a 1d nparray
        if self.threshold_method == "mean":
            return np.mean(feature)
        elif self.threshold_method == "median":
            return np.median(feature)
        elif self.threshold_method == "best":
            assert len(feature) == len(classes)
            classes = np.array(classes)
            data = np.zeros((len(feature), 2))
            data[:, 0] = feature
            data[:, 1] = classes
            data.sort(axis=0)
            unique_vals, unique_indices = np.unique(data[:, 0], return_index=True)
            best_threshold = 0
            best_alpha = float("-inf")
            for i in range(1, len(unique_vals)-1):
                negative_classes = list(data[:unique_indices[i], 1].astype(np.int))
                positive_classes = list(data[unique_indices[i]:, 1].astype(np.int))
                alpha = information_gain(classes, positive_classes) * float(len(positive_classes)) / len(classes) + \
                    information_gain(classes, negative_classes) * float(len(negative_classes)) / len(classes)
                if alpha > best_alpha:
                    best_threshold = unique_vals[i]
                    best_alpha = alpha
            return best_threshold

    def __most_freq_class__(self, classes):
        return (float(sum(classes)) / len(classes)) >= 0.5

    def __build_tree__(self, features, classes, depth=0):
        assert len(classes) > 0, "build tree called with no classes!"

        # Base case 1:
        # If all elements of a list are of the same class, return a leaf node with the appropriate class label.
        if len(classes) == 1:
            return DecisionNode(None, None, None, classes[0])
        else:
            all_same = True
            for i in range(1, len(classes)):
                if classes[i] != classes[0]:
                    all_same = False
                    break
            if all_same:
                return DecisionNode(None, None, None, classes[0])

        # Base case 2:
        # If a specified depth limit is reached, return a leaf labeled with the most frequent class.
        if depth == 0:
            most_freq_class = self.__most_freq_class__(classes)
            return DecisionNode(None, None, None, most_freq_class)

        # Features is a list of lists. Turn it into a numpy array.
        features = np.array(features)

        # For each attribute alpha: evaluate the normalized information gain gained by splitting on alpha
        best_feature = -1
        best_alpha = float("-inf")
        best_positive_classes = []
        best_negative_classes = []
        best_threshold = 0
        for feature in range(features.shape[1]):
            # Skip this feature if it isn't used
            if self.features_used != "all" and feature not in self.features_used:
                continue

            # For a continuous variable, find a cut threshold for splitting on the attribute
            # TODO: check if feature is binary?
            threshold = self.__find_cut_threshold__(features[:, feature], classes)

            positive_classes = []
            negative_classes = []
            for instance in range(features.shape[0]):
                if features[instance, feature] >= threshold:
                    positive_classes.append(classes[instance])
                else:
                    negative_classes.append(classes[instance])

            # if a pos/neg classes are zero length, we probably shouldn't split on this feature
            if len(positive_classes) == 0 or len(negative_classes) == 0:
                continue

            alpha = information_gain(classes, positive_classes) * float(len(positive_classes)) / len(classes) + \
                    information_gain(classes, negative_classes) * float(len(negative_classes)) / len(classes)
            if alpha > best_alpha:
                best_feature = feature
                best_alpha = alpha
                best_positive_classes = deepcopy(positive_classes)
                best_negative_classes = deepcopy(negative_classes)
                best_threshold = threshold

        # if this happens, just return a node with the most common class
        if best_feature == -1:
            most_freq_class = self.__most_freq_class__(classes)
            return DecisionNode(None, None, None, most_freq_class)

        # Create a decision node that splits on alpha_best
        dn = DecisionNode(None, None, lambda inst: inst[best_feature] >= best_threshold)

        # Recur on the sublists obtained by splitting on alpha_best, and add those nodes as children of node
        best_positive_features = features[features[:,best_feature] >= best_threshold, :]
        best_negative_features = features[features[:,best_feature] < best_threshold, :]
        dn.left = self.__build_tree__(best_positive_features, best_positive_classes, depth-1)
        dn.right = self.__build_tree__(best_negative_features, best_negative_classes, depth-1)
        return dn

    def classify(self, features):
        # Use a fitted tree to classify a list of feature vectors
        # Your output should be a list of class labels (either 0 or 1)
        classes = []
        for feature in features:
            classes.append(self.root.decide(feature))
        return classes


def load_csv(data_file_path, class_index=-1):
    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])
    classes= map(int,  out[:,class_index])
    features = out[:, :class_index]
    return features, classes


def generate_k_folds(dataset, k):
    # this method should return a list of folds,
    # where each fold is a tuple like (training_set, test_set)
    # where each set is a tuple like (examples, classes)
    features, classes = dataset
    assert(len(features) == len(classes))
    assert(k <= len(features))
    indices = range(0, len(features))
    random.shuffle(indices)

    folds = []
    fold_size = len(features)/k
    for fold_idx in range(k):
        # pick which entries out of the dataset go in the fold
        if fold_idx == k-1:
            test_indices = indices[fold_idx * fold_size :]
            train_indices = indices[: fold_idx * fold_size]
        else:
            test_indices = indices[fold_idx * fold_size : (fold_idx+1) * fold_size]
            train_indices = indices[:fold_idx * fold_size] + indices[(fold_idx+1) * fold_size:]

        # assemble the test set
        fold_examples = []
        fold_classes = []
        for i in test_indices:
            fold_examples.append(features[i])
            fold_classes.append(classes[i])
        test_set = (fold_examples, fold_classes)

        # assemble the train set
        fold_examples = []
        fold_classes = []
        for i in train_indices:
            fold_examples.append(features[i])
            fold_classes.append(classes[i])
        train_set = (fold_examples, fold_classes)

        folds.append((train_set, test_set))

    return folds


class RandomForest:

    def __init__(self, num_trees, depth_limit, example_subsample_rate, attr_subsample_rate):
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        assert len(features) == len(classes)

        #convert to numpy arrays
        features = np.array(features)
        classes = np.array(classes)

        for tree_num in range(self.num_trees):
            # Subsample the examples provided (with replacement) in accordance with a example subsampling rate.
            sample_example_indices = np.random.randint(0, features.shape[0], int(round(features.shape[0] * self.example_subsample_rate)))
            sample_features = features[sample_example_indices, :]
            sample_classes = list(classes[sample_example_indices])

            # Choose attributes at random to learn on, in accordance with an attribute subsampling rate.
            # TODO: with or without replacement?
            sample_attr_indices = random.sample(range(0,features.shape[1]), int(round(features.shape[1] * self.attr_subsample_rate)))  # without replacement
            #sample_attr_indices = np.random.randint(0, features.shape[1], int(round(features.shape[1] * self.attr_subsample_rate)))  # with replacement

            # Fit a decision tree to the subsample of data we've chosen (to a certain depth)
            self.trees.append(DecisionTree(depth_limit=self.depth_limit, features_used=set(sample_attr_indices)))
            self.trees[tree_num].fit(sample_features, sample_classes)
            #self.trees[tree_num].fit(features, classes)

    def __get_majority_vote__(self, classes):
        # take a majority vote of the classifications yielded by each tree
        #majority, _ = stats.mode(classes, axis=1)
        majority = ((np.sum(classes, axis=1).astype(np.float) / classes.shape[1]) >= 0.5).astype(np.int)
        return list(majority)

    def classify(self, features):
        # find classifications produced by each tree
        classes = np.zeros((len(features), self.num_trees), dtype=np.int)
        for tree_num in range(self.num_trees):
            classes[:, tree_num] = np.array(self.trees[tree_num].classify(features))
        return self.__get_majority_vote__(classes)


def ideal_parameters():
    ideal_depth_limit = 15
    ideal_esr = 0.8
    ideal_asr = 1
    return ideal_depth_limit, ideal_esr, ideal_asr


class ChallengeClassifier():

    def __init__(self, classifier="rf", **kwargs):
        self.classifier = classifier
        if kwargs == {}:
            self.kwargs = {"num_trees":21, "example_subsample_rate":0.3, "attr_subsample_rate":0.8, "depth_limit":10}
            #self.kwargs = {"depth_limit":10, "threshold_method":"mean"}
        else:
            self.kwargs = kwargs
        if self.classifier == "rf":
            self.classifier = RandomForest(**self.kwargs)
        else:
            self.classifier = DecisionTree(**self.kwargs)

    def fit(self, features, classes):
        self.classifier.fit(features, classes)

    def classify(self, features):
        return self.classifier.classify(features)

