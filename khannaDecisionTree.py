"""
Nikhil Khanna

This is an id3 decision tree designed to classify DNA sequences (see training.txt or validation.txt)
as to whether or not they are promoter sequences. Both a decision tree, and a boosted
ensemble of trees can be created, trained, and validated via this program
"""
from random import random
from bisect import bisect
from math import log

"""
This decision tree only accepts a fixed number of attributes (as it is designed to classify a particular file)
"""
NUMBER_ATTRIBUTES = 57
CHI_SQUARED_THRESHOLD = 3.5

"""
Helper function to determine if a training set consists of entirely promoters
or non-promoters
"""
def is_homogenous(training):
    if len(training) == 0:
        return True
    promoter_value = training[0].promoter
    for example in training:
        if example.promoter != promoter_value:
            return False
    return True

"""
Helper function to get the majority class (promoter or non-promoter) from a training set
"""
def get_majority_class(training):
    num_promoters = 0
    for example in training:
        if example.promoter:
            num_promoters += 1
    return num_promoters > len(training) / 2

"""
Runs a chi-squared-test based on splitting the training set on an attribute
to ensure the split is statistically significant enough so as not to overfit the training data
"""
def chi_squared_test(training, attribute_idx):
    p = 0
    n = 0
    for example in training:
        if example.promoter:
            p += 1.0
        else:
            n += 1.0
    partions = partion_data(training, attribute_idx)
    sum = 0
    for partion in partions:
        expected_p = float(p * (float(len(partion)) / float(len(training))))
        expected_n = float(float(len(partion)) - expected_p)
        actual_p = 0
        actual_n = 0
        for example in partion:
            if example.promoter:
                actual_p += 1.0
            else:
                actual_n += 1.0

        if expected_p > 0 and expected_n > 0:
            sum += (pow(actual_p - expected_p, 2) / expected_p) + (pow(actual_n - expected_n, 2) / expected_n)
    return sum >= CHI_SQUARED_THRESHOLD

"""
partions the training array into 4 different arrays based on the value of the attribute at the index
"""
def partion_data(training, idx):
    aList = []
    cList = []
    gList = []
    tList = []
    for example in training:
        if example.sequence[idx] == 'a':
            aList.append(example)
        elif example.sequence[idx] == 'c':
            cList.append(example)
        elif example.sequence[idx] == 'g':
            gList.append(example)
        else:
            tList.append(example)
    return [aList, cList, gList, tList]

"""
Function that returns the expected information required for a tree on the passed
in training set
"""
def training_I(training):
    p = 0
    n = 0
    for example in training:
        if example.promoter:
            p += 1
        else:
            n += 1
    return I(p, n)

"""
Helper function to compute information required for tree with the specified number
of promoter and non-promoter sequences
"""
def I(p, n):
    if p == 0 or n == 0:
        return 0
    p = float(p)
    n = float(n)
    return -(p/(p+n))*log(p/(p+n), 2) - (n/(p+n))*log(n/(p+n), 2)

"""
Function to compute information required for the tree that splits the training set on a particular index
"""
def E(training, attribute_idx):
    p = 0
    n = 0
    for example in training:
        if example.promoter:
            p += 1.0
        else:
            n += 1.0
    sum = 0
    partions = partion_data(training, attribute_idx)
    for partion in partions:
        pi = 0
        ni = 0
        for example in partion:
            if example.promoter:
                pi += 1.0
            else:
                ni += 1.0
        sum += ((pi + ni)/(p + n)) * I(pi, ni)
    return sum

"""
Computes the information gain via partioning the training set on a particular attribute
index (i.e. the nth letter in the nucleic acid sequence)
"""
def gain(training, attribute_idx):
    return training_I(training) - E(training, attribute_idx)

"""
Uses the metric of information gain to determine the best attribute idx
to split the tree on via the method used in the id3 decision tree paper
"""
def best_attribute_index(training):
    best_gain = 0
    best_attribute = 0
    for i in range(0, NUMBER_ATTRIBUTES):
        current_gain = gain(training, i)
        if current_gain > best_gain:
            best_gain = current_gain
            best_attribute = i
    return best_attribute

"""
Represents a dna sequence with both the nucleic acid sequence
(represented as a string with characters 'a', 'c', 'g', 't') and promoter status
(either true if the sequence is a promoter, or false otherwise)
"""
class DNASequence():
    def __init__(self, sequence, promoter):
        self.sequence = sequence
        self.promoter = (promoter == '+')

"""
A 4 way decision tree that can be constructed with a training set and then used
to classify sequences of nuclides (see training data for a template)
"""
class DecisionTree():

    """
    A non-leaf node in the decision tree, has an attribute_index representing
    which index it will be splitting on and an array of children nodes that result
    from that split
    """
    class DecisionTreeBodyNode():
        #map between a character and the appropriate index in the children array
        VALUE_INDEX_MAP = {'a': 0, 'c': 1, 'g': 2, 't': 3}

        def __init__(self, attribute_index):
            self.children = []
            self.isleaf = False
            self.attribute_index = attribute_index

        def get_child_node(self, attribute_value):
            return self.children[self.VALUE_INDEX_MAP[attribute_value]]

    """
    A leaf node in the decision tree. Simply has a value indicating whether
    it represents a promoter or not
    """
    class DecisionTreeLeaf():
        def __init__(self, promoter):
            self.isleaf = True
            self.promoter = promoter

    def __init__(self, training):
        self.root = self.construct_tree(training)

    """
    Constructs a 4-way decision tree (one branch at each non-leaf node for each character)
    based on the passed in training set
    """
    def construct_tree(self, training):
        #If the training set consists of entirely promoters or non-promoters, we place a leaf
        #with the appropriate value of promoter
        if is_homogenous(training):
            return self.DecisionTreeLeaf(training[0].promoter)

        best_attribute_idx = best_attribute_index(training)

        #To prevent overfitting, we pre-prune the tree by only splitting if the fit passes a chi-squared test
        #otherwise we cut off building the tree and place a leaf
        if not chi_squared_test(training, best_attribute_idx):
            return self.DecisionTreeLeaf(get_majority_class(training))

        #At this point we split the data on a particular index and recursively build the tree
        current_node = self.DecisionTreeBodyNode(best_attribute_idx)
        partions = partion_data(training, best_attribute_idx)
        for partion in partions:
            if len(partion) == 0:
                current_node.children.append(self.DecisionTreeLeaf(get_majority_class(training)))
            else:
                current_node.children.append(self.construct_tree(partion))
        return current_node

    def classify_sequence(self, seq):
        return self.classify_sequence_at_node(self.root, seq)

    """
    Classifies a sequence as a promoter or not through the decision tree
    """
    def classify_sequence_at_node(self, node, seq):
        if node.isleaf:
            return node.promoter
        character = seq[node.attribute_index]
        return self.classify_sequence_at_node(node.get_child_node(character), seq)

"""
A boosted ensemble of decision trees that can be constructed with a training set and then used
to classify sequences of nuclides (see training data for a template)
"""
class BoostedTree():
    #number of boosting rounds to use
    NUMBER_OF_ROUNDS = 75

    """
    Takes a training set and initializes a new BoostedTree
    """
    def __init__(self, training):
        self.trees = self.construct_boosted_tree(training)

    """
    Takes a training set and a parrelel array of the weights of each example and generates a weighted set
    """
    def weighted_bootstrap_set(self, training, weights):
        sample = []
        cum_weights = []
        total = 0.0
        for w in weights:
            total += w
            cum_weights.append(total)
        #randomly pick a training set in accordance with the passed in weights
        for i in range(0, len(training)):
            selectedIndex = bisect(cum_weights, total * random())
            sample.append(training[selectedIndex])
        return sample

    """
    Runs the boosting algorithm the speicifed number of times, and returns an array of tuples of
    decision trees and the weight of the "vote" of each tree
    """
    def construct_boosted_tree(self, training):
        weights = [1.0 / len(training)] * len(training)
        decisionTrees = []
        for roundIdx in range(0, self.NUMBER_OF_ROUNDS):
            #renomalizing the weights
            weights_sum = sum(weights)
            weights = map(lambda weight: weight / weights_sum, weights)
            tree = DecisionTree(self.weighted_bootstrap_set(training, weights))
            error = 0.0
            for i in range(0, len(training)):
                example = training[i]
                if example.promoter != tree.classify_sequence(example.sequence):
                    error += weights[i]
            if error > .5:
                return decisionTrees
            beta = error / (1 - error)
            #adjusting the weights to increase weight of misclassified examples
            for i in range(0, len(training)):
                example = training[i]
                if example.promoter == tree.classify_sequence(example.sequence):
                    weights[i] = weights[i] * beta
            decisionTrees.append((tree, beta))
        return decisionTrees

    """
    Classifes a sequence via weighted voting amongst the decision trees in this ensemble
    """
    def classify_sequence(self, sequence):
        trueVotes = 0.0
        falseVotes = 0.0
        for model in self.trees:
            if model[0].classify_sequence(sequence):
                trueVotes += 1.0 * log(1 / model[1])
            else:
                falseVotes += 1.0 * log(1/ model[1])
        return trueVotes >= falseVotes

"""
Reads in a file and returns an array of DNASequences representing the data in the file
"""
def get_sequences_from_file(file_name):
    with open(file_name) as f:
        dna_sequences = map(lambda line: DNASequence(line.split(" ")[0], line.split(" ")[1]), f.read().splitlines())
        return dna_sequences

"""
Runs a full accuracy test of the model on the passed in a validation set
"""
def run_accuracy_test(model, validation):
    num_correct = 0
    for example in validation:
        if model.classify_sequence(example.sequence) == example.promoter:
            num_correct += 1.0
    print num_correct / float(len(validation))

if __name__ == "__main__":
    training_sequences = get_sequences_from_file('training.txt')
    validation_sequences = get_sequences_from_file('validation.txt')
    tree = DecisionTree(training_sequences)
    boosted_tree = BoostedTree(training_sequences)
    print "training data accuracy with normal decision tree"
    run_accuracy_test(tree, training_sequences)
    print "validation data accuraccy with normal decision tree"
    run_accuracy_test(tree, validation_sequences)
    print "training data accuracy with boosted tree"
    run_accuracy_test(boosted_tree, training_sequences)
    print "validation data accuraccy with boosted tree tree"
    run_accuracy_test(boosted_tree, validation_sequences)
