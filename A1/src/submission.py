#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 1: binary classification
############################################################

############################################################
# Problem 1a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """

    # ### START CODE HERE ###
    parts = x.split()
    res = {} # default dict might be more natural here
    for p in parts:
        if p in res.keys():
            res[p] += 1
        else:
            res[p] = 1
    return res
    # ### END CODE HERE ###

############################################################
# Problem 1b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # ### START CODE HERE ###

    verbosity = 0

    allxs = [e[0] for e in trainExamples]
    allxString = " ".join(allxs)
    phi = featureExtractor(allxString)
    if not isinstance(phi, dict):
        raise ValueError("phi is not a dict but a {0}".format(type(phi)))

    weights = dict.fromkeys(phi.keys(), 0.0)

    def get_margin(weights, i):
        x, y = trainExamples[i]
        phi = featureExtractor(x)
        margin = dotProduct(weights, phi) * y
        return margin

    def zero_one_loss(weights, i):
        margin = get_margin(weights, i)
        loss = 1 if margin <= 0 else 0
        return loss

    def hinge_loss(weights, i):
        margin = get_margin(weights, i)
        loss = max(1 - margin, 0)
        return loss

    def myEval(examples, weights, lossFuns):
        res = 0.0
        losses = dict.fromkeys(lossFuns, 0.0)
        N = len(examples)
        for i in range(N):
            for k in lossFuns.keys():
                losses[k] += lossFuns[k](weights, i)

        for k in lossFuns.keys():
            losses[k] = float(losses[k]) / N
        return losses

    def hinge_loss_gradient(weights, i):
        x, y = trainExamples[i]
        phi = featureExtractor(x)
        val = dotProduct(weights, phi) * y
        if val < 1:
            res = {}
            for k in phi.keys():
                res[k] = -y*phi[k]
        else:
            res = dict.fromkeys(phi, 0.0)
        return res

    loss_funs = {"hinge": hinge_loss,
                "zero_one": zero_one_loss }

    for ei in range(numIters):
        for ti in range(len(trainExamples)):
            x, y = trainExamples[ti]
            phi = featureExtractor(x)

            gradient = hinge_loss_gradient(weights, ti)
            increment(d1=weights, scale=-eta, d2=gradient)

        if ei % 2 == 0:
            if verbosity > 0:
                print("end <{0}> total_seconds: {1}".format(dtend, dur))
                trainError = evaluatePredictor(trainExamples,
                                               lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
                testError = evaluatePredictor(testExamples,
                                             lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
                if testError > 0 or trainError > 0:
                    print("  errors: train: {0:.4f},  test: {1:.4f}".format(trainError, testError))
                trainLosses = myEval(trainExamples, weights, loss_funs)
                testLosses = myEval(testExamples, weights, loss_funs)
                for k in loss_funs.keys():
                    if trainLosses[k] > 0.0001:
                        print("  {0}, triain: {1:.4f}, test: {2:.4f}".format(k,
                                                                            trainLosses[k],
                                                                            testLosses[k]))

    # ### END CODE HERE ###
    return weights

############################################################
# Problem 1c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():

        # ### START CODE HERE ###

        phi = dict.fromkeys(weights, 0.0)
        for k in weights.keys():
            phi[k] = random.uniform(-1, 1)
        dot = dotProduct(weights, phi)
        y = int(dot > 0)
        # ### END CODE HERE ###
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 1e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # ### START CODE HERE ###
        verbosity = 0

        res = collections.defaultdict(lambda: 0)
        x_no_spaces = ''.join(x.split())
        xlen = len(x_no_spaces)
        for i in range(n, xlen+1, 1):
            res[x_no_spaces[i-n:i]] += 1
        return res
        # ### END CODE HERE ###
    return extract


def question_f():
    """1b-2-basic:  Test classifier on real polarity dev dataset."""
    trainExamples = readExamples('polarity.train')
    devExamples = readExamples('polarity.dev')
    for n in range(2, 12):
        print("==n: {0}".format(n))
        #featureExtractor = extractWordFeatures
        featureExtractor = extractCharacterFeatures(n)
        weights = learnPredictor(trainExamples, devExamples, featureExtractor, numIters=20, eta=0.01)
        outputWeights(weights, 'weights')
        outputErrorAnalysis(devExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
        trainError = evaluatePredictor(trainExamples,
                                       lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        devError = evaluatePredictor(devExamples,
                                     lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        print(("  Official: train error = %s, dev error = %s" % (trainError, devError)))


if __name__ == "__main__":
    if True:
        import cProfile
        cProfile.run('question_f()', 'submission.prof')