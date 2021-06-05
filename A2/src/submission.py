import shell
import util
import wordsegUtil

def func_name(i=1):
    import sys
    return sys._getframe(i).f_code.co_name

verbosity = 1
verbose = 1
############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # ### START CODE HERE ###

        if False:
            state = (self.query,)
            self.init_cost = sum([self.unigramCost(part) for part in state])
            self.end_state = "$" + self.query + '$'
            self.best_end = 10**8
        else:
            state = ((), self.query)
            self.init_cost = self.unigramCost(self.query)

        if self.query.startswith("zzzz"):
            print("Here")
        if verbosity > 0:
            cost = sum([self.unigramCost(part) for part in state])
            print("--Start--")
            print("Func: '{0}' state= {1}, cost: {2}".format(func_name(), state, self.init_cost), )
            self.best_cost = cost
        return state
        # ### END CODE HERE ###

    def isEnd(self, state):
        # ### START CODE HERE ###
        if False:
            if state[0] == self.end_state:
                return True
            else:
                return False
        else:
            if len(state[1]) == 0:
                return True
            else:
                return False


        # ### END CODE HERE ###

    def succAndCost(self, state):
        # ### START CODE HERE ###

        if verbosity > 1:
            cost = sum([self.unigramCost(word) for word in state])
            print("Func: '{0}' state= {1}, cost: {2}".format(func_name(), state, cost),)



        def segment_successors(state, uni_cost_func):
            done_part = list(state[0])

            next_part = state[1]
            assert isinstance(next_part, str)
            succ_states = []
            for i in range(1, len(next_part)+1):
                try:
                    new_word = next_part[:i]
                    new_right = next_part[i:]
                    left_part = done_part + [new_word]
                    new_state = (tuple(left_part), new_right)
                    action = i
                    tran_cost = uni_cost_func(new_word)
                    succ_states.append((action, new_state, tran_cost))
                except Exception as e:
                    print(e)
            return succ_states

        def segment_successors_old(words):
            cur_cost = sum([self.unigramCost(word) for word in words])
            succ_states = []
            for wi in range(len(words)):
                word = words[wi]
                if len(word) <= 1:
                    return []
                old_word_cost = self.unigramCost(word)
                for sj in range(1, len(word)):
                    action = (wi, sj,) # add space after letter j in word wi
                    new_left = word[:sj]
                    new_right = word[sj:]
                    new_words_cost = self.unigramCost(new_left) + self.unigramCost(new_right)
                    new_words = (new_left, new_right,)
                    new_state = tuple(words[:wi] + new_words + words[wi + 1:], )
                    new_state_cost = cur_cost - old_word_cost + new_words_cost
                    cost_change = new_state_cost - cur_cost
                    succ_states.append((action, new_state, cost_change,))
            if state[0] != self.query:
                succ_states.append(((0, 0), (self.end_state,), 0,))
            return succ_states

        my_cost = sum([self.unigramCost(w) for w in state[0]])
        if my_cost > self.init_cost:
            return []

        next_states = segment_successors(state=state, uni_cost_func=self.unigramCost)

        return next_states
        # ### END CODE HERE ###


def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    if query.startswith("zzzzz"):
        print(query)
    ucs = util.UniformCostSearch(verbose=verbose)
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

    state = [[],query]
    for ai in ucs.actions:
        try:
            left_words, right_word = state
            new_word = right_word[:ai]
            left_words.append(new_word)
            new_right = right_word[ai:]
            state = [left_words, new_right]
        except Exception as e:
            print(e)
        if len(new_right) == 0:
            break

    left_words, right_word = state
    res = ' '.join(left_words)
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
        state = (self.queryWords,)

        self.end_state = ("--END-State--",)

        if verbosity > 0:
            print("--Start--")
            pairs = zip(self.queryWords, self.queryWords[1:])
            cost = sum([self.bigramCost(a, b) for a, b in pairs])
            if len(self.queryWords) > 0:
                cost += self.bigramCost(wordsegUtil.SENTENCE_BEGIN, self.queryWords[0])
            print("Func: '{0}' state= {1}, cost: {2}".format(func_name(), state, cost,),)

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

        def apply_bigram_cost(words):
            cost = 0
            if len(words) > 0:
                cost += self.bigramCost(wordsegUtil.SENTENCE_BEGIN, words[0])
            pairs = zip(words, words[1:])
            cost += sum([self.bigramCost(a, b) for a, b in pairs])
            return cost

        def replace_successors(state, apply_bigram_cost):
            cur_words = list(state)
            cur_cost = apply_bigram_cost(cur_words)
            if verbosity > 1:
                print("Func: '{0}' state= {1}, cur_cost: {2}".format(func_name(), state, cur_cost),)
            succ_states = []
            for wi, old_word in enumerate(cur_words):
                try:
                    replace_words = self.possibleFills(old_word)
                except Exception as e:
                    pass
                    replace_words = []
                for replace_word in replace_words:
                    new_words = cur_words.copy()
                    action = (wi, (old_word, replace_word),)
                    new_words[wi] = replace_word

                    replace_cost = apply_bigram_cost(new_words)
                    cost_change = replace_cost - cur_cost

            return succ_states

        succ_states = replace_successors(state=state, apply_bigram_cost=apply_bigram_cost)
        return succ_states

        # ### END CODE HERE ###


def insertVowels(queryWords, bigramCost, possibleFills):
    # ### START CODE HERE ###
    ucs = util.UniformCostSearch(verbose=verbose)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))

    words = queryWords

    if ucs.actions is None or len(ucs.actions) == 0:
        return ' '.join(queryWords)

    for wi, (old_word, new_word) in ucs.actions:
        if words[wi] != old_word:
            msg = "error expecting {0} at pos {1} but found {2} in {3}".format(old_word, wi,
                                                                               words[wi], words)
            raise ValueError(msg)
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
        # ### START CODE HERE ###
        state = (self.query,)
        self.end_state = ("$END-State$",)

        def apply_bigram_cost(words):
            cost = 0
            if len(words) > 0:
                cost += self.bigramCost(wordsegUtil.SENTENCE_BEGIN, words[0])
            pairs = zip(words, words[1:])
            cost += sum([self.bigramCost(a, b) for a, b in pairs])
            return cost

        self.init_cost = apply_bigram_cost(self.query)
        self.best_end = self.init_cost
        if verbosity > 0:
            print("\n\n--Start 3--")
            print(" -->>{0} state= {1} init_cost: {2}  best_end: {3}".format(func_name(),
                                                                             state,
                                                                             self.init_cost,
                                                                             self.best_end),)

        return state
        # ### END CODE HERE ###

    def isEnd(self, state):
        # ### START CODE HERE ###
        if state[0] == self.end_state:
            if verbosity > 0:
                print("  -->> {0} end-state= {1}".format(func_name(), state,), )
            res = True
        else:
            res = False
        return res
        # ### END CODE HERE ###

    def succAndCost(self, state):

        # ### START CODE HERE ###
        words = state

        if verbosity > 1:
            print("\n-->> '{0}' state= {1}, words: {2} ".format(func_name(1), state, words))
        cur_words = list(words)
        real_words = cur_words.copy()
        for i in range(len(cur_words)):
            if real_words[i][0] == "$":
                real_words[i] = real_words[i][1:]

        pairs = zip(cur_words, real_words[1:])
        state_cost = sum([self.bigramCost(a, b) for a, b in pairs])
        if len(cur_words) > 0:
            state_cost += self.bigramCost(wordsegUtil.SENTENCE_BEGIN, real_words[0])

        def apply_bigram_cost(words):
            cost = 0
            if len(words) > 0:
                cost += self.bigramCost(wordsegUtil.SENTENCE_BEGIN, words[0])
            pairs = zip(words, words[1:])
            cost += sum([self.bigramCost(a, b) for a, b in pairs])
            return cost

        def segment_successors(words, apply_bigram_cost):
            cur_cost = apply_bigram_cost(words)
            if verbosity > 1:
                print("   -> {0} state= {1}, cur_cost: {2}".format(func_name(), state, cur_cost),)
            succ_states = []
            for wi in range(len(words)):
                word = words[wi]
                if word[0] == "$":
                    continue
                if False:
                    if len(word) <= 1:
                        return []
                for sj in range(1, len(word)):
                    action = ('S', wi, sj,) # add space after letter j in word wi
                    new_left = word[:sj]
                    new_right = word[sj:]
                    new_words = tuple(words[:wi] + [new_left, new_right,] + words[wi + 1:], )
                    new_state_cost = apply_bigram_cost(new_words)
                    cost_change = new_state_cost - cur_cost
                    succ_states.append((action, new_words, 0,))
            return succ_states

        def replace_successors(words, apply_bigram_cost):
            cur_words = list(state)
            cur_cost = apply_bigram_cost(cur_words)
            if verbosity > 1:
                print("   -> {0} state= {1}, cur_cost: {2}".format(func_name(), state, cur_cost),)
            succ_states = []
            for wi, old_word in enumerate(cur_words):
                if old_word[0] == "$":
                    continue
                try:
                    replace_words = self.possibleFills(old_word)
                except Exception as e:
                    pass
                    replace_words = []
                for replace_word in replace_words:
                    new_words = cur_words.copy()
                    new_words[wi] = replace_word
                    replace_cost = apply_bigram_cost(new_words)
                    cost_change = replace_cost - cur_cost
                    if replace_cost < cur_cost:
                        new_words[wi] = "$" + replace_word
                        action = ("R", wi, (old_word, replace_word),)
                        new_state = (action, tuple(new_words,), 0)
                        succ_states.append(new_state)
            if verbosity > 1:
                print("  succ_states: [{0}]".format(succ_states),)
            return succ_states

        segment_states = segment_successors(cur_words, apply_bigram_cost)
        replace_states = replace_successors(cur_words, apply_bigram_cost)
        if False:
            for action, state, cost in segment_states:
                rstates = replace_successors(state, apply_bigram_cost)
                replace_states += rstates

        next_states = replace_states + segment_states

        if cur_words[0] != self.query and cur_words[0] != self.end_state:
            if state_cost < self.best_end:
                self.best_end = state_cost
                if verbosity > 0:
                    print("** adding link to end state from {0} with cost {1} cur_words {2}".format(state, state_cost, cur_words),)
                next_states.append(( ('N', 0, 0), self.end_state, state_cost,))


        if verbosity > 0.9:
            print("next_states: ")
            for i, ns in enumerate(next_states):
                print("{0}: {1}".format(i, ns))
                action, state, cost = ns
                print("[{0}]: action: {1}, state{2}, cost: {3}".format(i, action, state, cost))
        return next_states
        # ### END CODE HERE ###

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # ### START CODE HERE ###
    ucs = util.UniformCostSearch(verbose=verbose)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))


    words = [query,]
    pairs = zip(words, words[1:])
    init_cost = sum([bigramCost(a, b) for a, b in pairs])
    if len(words) > 0:
        init_cost += bigramCost(wordsegUtil.SENTENCE_BEGIN, words[0])
    if verbosity > 0:
        print("Func: '{0}' query= {1}, init_cost: {2}".format(func_name(), query, init_cost),)
        print(" DONE: totalCost: {0}, actions: {1}".format(ucs.totalCost, ucs.actions))
        print("  NumStatesExplored: {0}".format(ucs.numStatesExplored,))

    if ucs.totalCost is None:
        return query

    #if ucs.totalCost >= init_cost:
    #    return query

    if ucs.actions is None or len(ucs.actions) == 0:
        return query

    for ai, (atype, wi, ainfo) in enumerate(ucs.actions):
        if verbosity > 0:
            print("ai: {0}, atype: {1}, wi: {2}, ainfo: {3}".format(ai, atype, wi, ainfo))
        if atype == 'R':
            old_word, new_word = ainfo
            if words[wi] != old_word:
                msg = "error expecting {0} at pos {1} but found {2} in {3}".format(old_word, wi,words[wi], words)
                raise ValueError(msg)
            words[wi] = new_word
        elif atype == 'S':
            word = words[wi]
            new_left = word[:ainfo]
            new_right = word[ainfo:]
            words = words[:wi] + [new_left, new_right, ] + words[wi + 1:]
        elif atype == 'N':
            continue
        else:
            raise RuntimeError("unexpected atype: {0}, expected 'N', 'R' or 'S'")
    res = ' '.join(words)
    if verbosity > 0:
        print(" **answer: {0}, joined: {1}".format(words, res))
        print("**res: {0}, cost: {1}".format(res, ucs.totalCost))
        print("")
    return res
    # ### END CODE HERE ###

############################################################


if __name__ == '__main__':

    shell.main()
