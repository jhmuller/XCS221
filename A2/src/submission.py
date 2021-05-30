import shell
import util
import wordsegUtil

def func_name(i=1):
    import sys
    return sys._getframe(1).f_code.co_name
verbosity = 0
############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # ### START CODE HERE ###
        state = (self.query,)
        self.init_cost = sum([self.unigramCost(part) for part in state])
        self.end_state = "$" + self.query + '$'
        self.best_end = 10**8
        if verbosity > 0:
            cost = sum([self.unigramCost(part) for part in state])
            print("--Start--")
            print("Func: '{0}' state= {1}, cost: {2}".format(func_name(), state, cost),)
            self.best_cost = cost
        return state
        # ### END CODE HERE ###

    def isEnd(self, state):
        # ### START CODE HERE ###
        if state[0] == self.end_state:
            return True
        else:
            return False
        return True

        # ### END CODE HERE ###

    def succAndCost(self, state):
        # ### START CODE HERE ###
        if state[0] == self.end_state:
            return []

        if verbosity > 0:
            cost = sum([self.unigramCost(part) for part in state])
            print("Func: '{0}' state= {1}, cost: {2}".format(func_name(), state, cost),)
        next_states = []

        parts = state
        # cost_to_here = state[1]
        cur_cost = sum([self.unigramCost(part) for part in parts])
        if verbosity > 0:
            if cur_cost < self.best_end:
                print("  adding end state cost= {0}".format(cur_cost))
                self.best_end = cur_cost

        if state[0] == self.query:
            pass
            # print("  state: {0}".format(state))
            # next_states.append(((0, 0), (self.end_state,), cur_cost,))
        else:
            next_states.append(((0, 0), (self.end_state,), 0,))
        for pi in range(len(parts)):
            part = parts[pi]
            old_part_cost = self.unigramCost(part)
            if len(part) > 1:
                for j in range(1, len(part)):
                    action = (pi, j,)
                    new_left = part[:j]
                    new_right = part[j:]
                    new_parts_cost = self.unigramCost(new_left) + self.unigramCost(new_right)
                    new_parts = (new_left, new_right,)
                    new_state = tuple(parts[:pi] + new_parts + parts[pi+1:],)
                    new_state_cost = cur_cost - old_part_cost + new_parts_cost
                    cost_change = new_state_cost - cur_cost
                    next_states.append((action, new_state, cost_change,))

                    if verbosity > 0:
                        print("\n  cur_state: {0}, cur_cost: {1}".format(state, cur_cost))
                        print("  old_part: {0}, old_part_cost: {1}".format(part, old_part_cost))
                        print("  new_parts: {0}, new_parts_cost {1}".format(new_parts, new_parts_cost))
                        print("  adding state {0} with cost_change {1}\n".format(new_state, cost_change))
        if verbosity > 0:
            pass
            # ("  next_states: {0}".format(next_states))
        return next_states
        # ### END CODE HERE ###


def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # ### START CODE HERE ###
    if verbosity > 0:
        print("totalCost: {0}".format(ucs.totalCost))
        print("actions: {0}".format(ucs.actions))
        print("numStatesExplored: {0}".format(ucs.numStatesExplored))

    query_cost = unigramCost(query)
    if ucs.totalCost >= query_cost:
        return query

    if ucs.actions is None:
        return query

    parts = [query]
    for pi, j in ucs.actions:
        if j == 0:
            continue
        part = parts[pi]
        new_left = part[:j]
        new_right = part[j:]
        parts = parts[:pi] + [new_left, new_right,] + parts[pi + 1:]
    res = ' '.join(parts)
    if verbosity > 0:
        print(" res: '{0}'".format(res))
        print("--End--\n")
    return res
    # ### END CODE HERE ###

############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # ### START CODE HERE ###
        state = (tuple(self.queryWords,))

        self.end_state = ("--END-State--",)

        if verbosity > 0:
            print("--Start--")
            pairs = zip(self.queryWords, self.queryWords[1:])
            cost = sum([self.bigramCost(a, b) for a, b in pairs])
            if len(self.queryWords) > 0:
                cost += self.bigramCost(wordsegUtil.SENTENCE_BEGIN, self.queryWords[0])
            print("Func: '{0}' state= {1}, cost: {2} possibleFills: {3}".format(func_name(),
                                                                                state,
                                                                                cost,
                                                                                self.possibleFills),)

        return state
        # ### END CODE HERE ###

    def isEnd(self, state):
        # ### START CODE HERE ###

        next_states = self.succAndCost(state)
        if len(next_states) == 0:
            return True
        else:
            return False
        # ### END CODE HERE ###

    def succAndCost(self, state):
        # ### START CODE HERE ###
        cur_words = list(state)
        pairs = zip(cur_words, cur_words[1:])
        cur_cost = sum([self.bigramCost(a, b) for a, b in pairs])
        if len(self.queryWords) > 0:
            cur_cost += self.bigramCost(wordsegUtil.SENTENCE_BEGIN, cur_words[0])

        if verbosity > 0:
            print("Func: '{0}' state= {1}, cur_cost: {2}".format(func_name(), state, cur_cost),)
        next_states = []
        for wi, word in enumerate(cur_words):
            # print("  possibleFills: {0}, type: {1}".format(self.possibleFills,
            #                                               type(self.possibleFills)))
            try:
                replace_words = self.possibleFills(word)
            except Exception as e:
                if False:
                    print(" trying to get replacement for {0}".format(word))
                    print(" state: {0}".format(state))
                    print(" self.query_words: {0}".format(self.queryWords))
                    print(e)
                replace_words = []
                #raise ValueError(e)
            # print("  possibleFills for {0}, {1}".format(word, replace_words))
            for replace_word in replace_words:
                new_words = cur_words.copy()
                action = (wi, replace_word,)
                new_words[wi] = replace_word
                pairs = zip(new_words[:-1], new_words[1:])

                replace_cost = sum([self.bigramCost(a, b) for a, b in pairs])
                replace_cost += self.bigramCost(wordsegUtil.SENTENCE_BEGIN, new_words[0])
                # print(" replace_cost: {0}".format(replace_cost))
                cost_change = replace_cost - cur_cost
                new_state = (action, tuple(new_words,), cost_change)
                next_states.append(new_state)
        return next_states
        # ### END CODE HERE ###

def insertVowels(queryWords, bigramCost, possibleFills):
    # ### START CODE HERE ###
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))

    words = queryWords

    if ucs.actions is None or len(ucs.actions) == 0:
        return ' '.join(queryWords)

    for wi, new_word in ucs.actions:
        words[wi] = new_word
    res = ' '.join(words)
    if verbosity > 0:
        print("**res: {0}, cost: {1}".format(res, ucs.totalCost))
        print("")
    return res
    # ### END CODE HERE ###

############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        pass
        # ### START CODE HERE ###
        # ### END CODE HERE ###

    def isEnd(self, state):
        pass
        # ### START CODE HERE ###
        # ### END CODE HERE ###

    def succAndCost(self, state):
        pass
        # ### START CODE HERE ###
        # ### END CODE HERE ###

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # ### START CODE HERE ###
    # ### END CODE HERE ###

############################################################


if __name__ == '__main__':

    import grader
    Test1b = grader.Test_1b()
    #Test1b.test_0()
    Test1b.test_1()
    Test1b.test_2()

    shell.main()
