import decision_tree as dt
import numpy as np
import sys
import pickle
import random
import math


def optimize_rf_params(dataset, num_folds):
    print "running k-fold CV on classifier"

    folds = dt.generate_k_folds(dataset, num_folds)

    # RF params
    num_trees = 5
    depth_limit = [15,16,17]
    example_subsample_rate = [0.8]
    attr_subsample_rate = [1]
    #depth_limit = [float("inf")]
    #example_subsample_rate = np.arange(0.1, 1, 0.1)
    #attr_subsample_rate = np.arange(0.2, 1.2, 0.2)

    avg_accuracies = []
    for dl in depth_limit:
        for esr in example_subsample_rate:
            for asr in attr_subsample_rate:
                accuracies = []
                for fold in folds:
                    train, test = fold
                    train_features, train_classes = train
                    test_features, test_classes = test
                    rf = dt.RandomForest(num_trees, dl, esr, asr)
                    rf.fit(train_features, train_classes)
                    output = rf.classify(test_features)

                    accuracies.append( dt.accuracy(output, test_classes))
                    #precisions.append( dt.precision(output, test_classes))
                    #recalls.append( dt.recall(output, test_classes))
                    #confusion.append( dt.confusion_matrix(output, test_classes))
                sys.stdout.write('.')
                avg_accuracy = str(sum(accuracies) / len(accuracies))
                avg_accuracies.append((avg_accuracy, {"num_trees" : num_trees, "dl" : dl, "esr" : esr, "asr" : asr}))

    avg_accuracies = sorted(avg_accuracies, reverse=True)
    print "\nbest accuracy: " + str(avg_accuracies[0][0])
    for k,v in avg_accuracies[0][1].iteritems():
        print k + " = " + str(v)


def optimize_challenge_classifier_params(dataset, num_folds):
    print "running k-fold CV on challenge classifier"

    # cut down dataset a bit
    features, classes = dataset
    #sample = random.sample(xrange(len(features)), 10000)
    #features = [features[i] for i in sample]
    #classes = [classes[i] for i in sample]
    #dataset = (features, classes)

    folds = dt.generate_k_folds(dataset, num_folds)

    # RF params
    num_trees = 21
    depth_limit = [10]
    example_subsample_rate = [0.3]
    #attr_subsample_rate = [math.sqrt(len(features[0]))/len(features[0])]
    attr_subsample_rate = [0.8]
    #depth_limit = np.arange(1, 8, 1)
    #example_subsample_rate = np.arange(0.1, 1, 0.1)
    #attr_subsample_rate = np.arange(0.2, 1, 0.2)

    avg_accuracies = []
    for dl in depth_limit:
        for esr in example_subsample_rate:
            for asr in attr_subsample_rate:
                accuracies = []
                for fold in folds:
                    train, test = fold
                    train_features, train_classes = train
                    test_features, test_classes = test
                    #rf = dt.ChallengeClassifier()
                    #rf = dt.ChallengeClassifier("dt", depth_limit=10)
                    rf = dt.ChallengeClassifier("rf", num_trees=num_trees, example_subsample_rate=esr, attr_subsample_rate=asr, depth_limit=dl)
                    rf.fit(train_features, train_classes)
                    output = rf.classify(test_features)

                    accuracies.append( dt.accuracy(output, test_classes))
                    #precisions.append( dt.precision(output, test_classes))
                    #recalls.append( dt.recall(output, test_classes))
                    #confusion.append( dt.confusion_matrix(output, test_classes))
                sys.stdout.write('.')
                avg_accuracy = str(sum(accuracies) / len(accuracies))
                avg_accuracies.append((avg_accuracy, {"num_trees" : num_trees, "dl" : dl, "esr" : esr, "asr" : asr}))

    avg_accuracies = sorted(avg_accuracies, reverse=True)
    print "\nbest accuracy: " + str(avg_accuracies[0][0])
    for k,v in avg_accuracies[0][1].iteritems():
        print k + " = " + str(v)

if __name__ == "__main__":
    #dataset = dt.load_csv('part2_data.csv')
    #optimize_rf_params(dataset, 10)
    dataset = pickle.load(open("challenge_data.pickle", "rb"))
    optimize_challenge_classifier_params(dataset, 10)