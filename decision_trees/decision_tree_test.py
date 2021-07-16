import unittest

import decision_tree as dt
import numpy as np
import pickle


class TestVerification(unittest.TestCase):

    def setUp(self):
        self.classifier_output = [0,1,1,0,0,0,1,1,1,1]
        self.classes           = [0,1,1,1,1,1,0,0,0,0]

    def test_confusion_matrix(self):
        self.assertEqual(dt.confusion_matrix(self.classifier_output, self.classes), [[2,3],[4,1]])

    def test_precision(self):
        self.assertAlmostEqual(dt.precision(self.classifier_output, self.classes), 0.3333, places=3)

    def test_recall(self):
        # recall is measured as: true_positive/ (true_positive + false_negative)
        self.assertAlmostEqual(dt.recall(self.classifier_output, self.classes), 0.4, places=3)

    def test_accuracy(self):
        # accuracy is measured as:  correct_classifications / total_number_examples
        self.assertAlmostEqual(dt.accuracy(self.classifier_output, self.classes), 0.3, places=3)


class TestEntropy(unittest.TestCase):
    pass
    # def entropy(class_vector):
    #     # TODO: Compute the Shannon entropy for a vector of classes
    #     # Note: Classes will be given as either a 0 or a 1.
    #     raise NotImplemented()
    #
    # def test_information_gain():
    #     """ Assumes information_gain() accepts (classes, [list of subclasses])
    #         Feel free to edit / enhance this note with more tests """
    #     restaurants = [0]*6 + [1]*6
    #     split_patrons =   [[0,0], [1,1,1,1], [1,1,0,0,0,0]]
    #     split_food_type = [[0,1],[0,1],[0,0,1,1],[0,0,1,1]]
    #     gain_patrons = information_gain(restaurants, split_patrons)
    #     gain_type = information_gain(restaurants, split_food_type)
    #     assert round(gain_patrons,3) == 0.541, "Information Gain on patrons should be 0.541"
    #     assert gain_type == 0.0, "Information gain on type should be 0.0"
    #     print "Information Gain calculations correct..."


class TestManualDecisionTree(unittest.TestCase):

    def setUp(self):
        self.examples = [[1,0,0,0],
            [1,0,1,1],
            [0,1,0,0],
            [0,1,1,0],
            [1,1,0,1],
            [0,1,0,1],
            [0,0,1,1],
            [0,0,1,0]]

        self.classes = [1,1,1,0,1,0,1,0]
        self.decision_tree_root = dt.build_decision_tree()
        self.classifier_output = [self.decision_tree_root.decide(example) for example in self.examples]

    def test_build_decision_tree(self):
        # Make sure your hand-built tree is 100% accurate.
        p1_accuracy = dt.accuracy( self.classifier_output, self.classes )
        p1_precision = dt.precision(self.classifier_output, self.classes)
        p1_recall = dt.recall(self.classifier_output, self.classes)
        p1_confusion_matrix = dt.confusion_matrix(self.classifier_output, self.classes)

        self.assertEqual(p1_accuracy, 1)
        self.assertEqual(p1_precision, 1)
        self.assertEqual(p1_recall, 1)
        self.assertEqual(p1_confusion_matrix, [[5, 0], [0, 3]])


class TestDecisionTree(unittest.TestCase):

    def setUp(self):
        pass

    def test_build_tree_basecase1(self):
        tree = dt.DecisionTree()
        features = \
            [[0,1,0,1],
             [1,0,1,1],
             [1,1,0,0]]
        classes = [1,1,1]
        root = tree.__build_tree__(features, classes, 100)
        self.assertEqual(root.class_label, 1)
        classes = [0,0,0]
        root = tree.__build_tree__(features, classes, 100)
        self.assertEqual(root.class_label, 0)

    def test_build_tree_basecase2(self):
        tree = dt.DecisionTree()
        features = \
            [[0,1,0,1],
             [1,0,1,1],
             [1,1,0,0]]
        classes = [1,0,0]
        root = tree.__build_tree__(features, classes, 0)
        self.assertEqual(root.class_label, 0)
        classes = [1,0,1]
        root = tree.__build_tree__(features, classes, 0)
        self.assertEqual(root.class_label, 1)

    def test_decision_tree_banknote(self):
        print "running 10-fold CV on DT classifier using banknote authentication dataset"
        dataset = dt.load_csv('part2_data.csv')
        ten_folds = dt.generate_k_folds(dataset, 10)

        # on average your accuracy should be higher than 60%.
        accuracies = []
        precisions = []
        recalls = []
        confusion = []

        for fold in ten_folds:
            train, test = fold
            train_features, train_classes = train
            test_features, test_classes = test
            tree = dt.DecisionTree( )
            tree.fit( train_features, train_classes)
            output = tree.classify(test_features)

            accuracies.append( dt.accuracy(output, test_classes))
            precisions.append( dt.precision(output, test_classes))
            recalls.append( dt.recall(output, test_classes))
            confusion.append( dt.confusion_matrix(output, test_classes))

        print "average accuracy: " + str(sum(accuracies) / len(accuracies))
        print "average precision: " + str(sum(precisions) / len(precisions))
        print "average recall: " + str(sum(recalls) / len(recalls))

        self.assertGreaterEqual(str(sum(accuracies) / len(accuracies)), 0.6)


class TestRandomForest(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_majority_vote(self):
        rf = dt.RandomForest(5, 5, 0.5, 0.5)
        classes = np.array([[1, 1, 1, 0, 0],
                            [1, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1]])
        votes = rf.__get_majority_vote__(classes)
        self.assertEqual(votes, [1, 0, 0, 1])

    def test_random_forest_banknote(self):
        print "running 10-fold CV on RF classifier using banknote authentication dataset"
        dataset = dt.load_csv('part2_data.csv')
        ten_folds = dt.generate_k_folds(dataset, 10)

        # on average your accuracy should be higher than 60%.
        accuracies = []
        precisions = []
        recalls = []
        confusion = []

        # RF params
        num_trees = 5
        depth_limit = 15
        example_subsample_rate = 0.8
        attr_subsample_rate = 1

        for fold in ten_folds:
            train, test = fold
            train_features, train_classes = train
            test_features, test_classes = test
            rf = dt.RandomForest(num_trees, depth_limit, example_subsample_rate, attr_subsample_rate)
            rf.fit(train_features, train_classes)
            output = rf.classify(test_features)

            accuracies.append( dt.accuracy(output, test_classes))
            precisions.append( dt.precision(output, test_classes))
            recalls.append( dt.recall(output, test_classes))
            confusion.append( dt.confusion_matrix(output, test_classes))

        print "average accuracy: " + str(sum(accuracies) / len(accuracies))
        print "average precision: " + str(sum(precisions) / len(precisions))
        print "average recall: " + str(sum(recalls) / len(recalls))

        self.assertGreaterEqual(str(sum(accuracies) / len(accuracies)), 0.75)


class TestChallengeClassifier(unittest.TestCase):
    def setUp(self):
        self.dataset = pickle.load(open("challenge_data.pickle", "rb"))

    def test_challenge_classifier_default(self):
        print "Testing challenge classifier with default params (10-fold CV, repeated 10 times)"
        accuracies = []
        for i in range(1):
            self.folds = dt.generate_k_folds(self.dataset, 10)
            for fold in self.folds:
                train, test = fold
                train_features, train_classes = train
                test_features, test_classes = test
                cc = dt.ChallengeClassifier()
                cc.fit(train_features, train_classes)
                output = cc.classify(test_features)
                accuracies.append( dt.accuracy(output, test_classes))

        avg_accuracy = str(sum(accuracies) / len(accuracies))
        print "average accuracy: " + str(avg_accuracy)