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
    import datetime
    verbosity = 0

    def whoami(i=1):
        return sys._getframe(i).f_code.co_name

    parent = whoami(2)
    if parent == "test_2":
        print("called from {0}, <{1}>".format(parent, datetime.datetime.now()))
        verbosity = 1
    from collections import defaultdict
    argdict = locals().copy()
    if verbosity > 0:
        print(whoami(2))

    parts = x.split()
    res = defaultdict(lambda: 0)
    for p in parts:
        res[p] += 1
    # print("input: {0}, output: {1}".format(x, res))
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
    import datetime
    verbosity = 0

    def whoami(i=1):
        return sys._getframe(i).f_code.co_name

    parent = whoami(2)
    if parent == "test_2":
        print("called from {0}, <{1}>".format(parent, datetime.datetime.now()))
        verbosity = 1

    if verbosity > 0:
        argdict = locals().copy()
        print(whoami())
        if verbosity > 1:
            print("  Args")
            for k in argdict.keys():
                print("  {0}: {1}".format(k, argdict[k]))

    allxs = [e[0] for e in trainExamples]
    allxString = " ".join(allxs)
    phi = featureExtractor(allxString)
    if not isinstance(phi, dict):
        raise ValueError("phi is not a dict but a {0}".format(type(phi)))

    weights = dict.fromkeys(phi.keys(), 0.0)
    if verbosity > 1:
        print("weights: {0}".format(weights))

    def dictdot(xdict, ydict):
        argdict = locals().copy()
        if verbosity > 2:
            print(whoami())
            if verbosity > 1:
                print("--Args")
                for k in argdict.keys():
                    print("  {0}: {1}".format(k, argdict[k]))
        res = 0
        for k in xdict.keys():
            res += xdict[k] * ydict[k]
        if verbosity > 2:
            print(" res: {0}".format(res))
        return res

    # For stochastic gradient descent
    def sFtrain(w, i):
        argdict = locals().copy()
        if verbosity > 2:
            print(whoami())
            if verbosity > 1:
                print("  Args")
                for k in argdict.keys():
                    print("  {0}: {1}".format(k, argdict[k]))
        x, y = trainExamples[i]
        phi = featureExtractor(x)
        res = (dictdot(w, phi) - y) ** 2
        return res

    def sdFtrain(w, i):
        argdict = locals().copy()
        if verbosity > 1:
            print(whoami())
            if verbosity > 1:
                print("  Args")
                for k in argdict.keys():
                    print("  {0}: {1}".format(k, argdict[k]))
        x, y = trainExamples[i]
        phi = featureExtractor(x)
        mult = 2 * (dictdot(w, phi) - y)

        res = phi.copy()
        for k in res.keys():
            res[k] *= mult
        if verbosity > 1:
            print("  {0} res: {0}".format(whoami(), res))
        return res

    numUpdates = 0
    for ni in range(numIters):
        for ti in range(len(trainExamples)):
            x, y = trainExamples[ti]
            if verbosity > 0:
                if ti % 10000 == 0:
                    print("iter: {0}, ti: {1}, len(x): {2} y: {3} <{4}>".format(ni, ti, len(x), y, datetime.datetime.now()))
            #value = sFtrain(weights, i)
            gradient = sdFtrain(weights, ti)
            numUpdates += 1
            eta = 1.0 / numUpdates  # Remember to do 1.0 instead of 1!
            for k in weights.keys():
                weights[k] = weights[k] - eta * gradient[k]
        if verbosity > 0:
            print('iteration {}: w = {}'.format(ni, weights))



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
        def dictdot(xdict, ydict):
            res = 0
            for k in xdict.keys():
                res += xdict[k] * ydict[k]
            return res

        phi = dict.fromkeys(weights, 0.0)
        for k in weights.keys():
            phi[k] = random.uniform(-1, 1)
        dot = dictdot(weights, phi)
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
        pass
        # ### START CODE HERE ###
        extract = collections.defaultdict(lambda: 0)
        xlen = len(x)
        for i in range(n, xlen, 1):
            extract[x[i-n:i]] += 1
        # ### END CODE HERE ###
    return extract
