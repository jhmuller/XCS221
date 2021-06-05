import shell
import util
import wordsegUtil


############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # ### START CODE HERE ###
        self.verbosity = 1
        state = 0
        if self.verbosity > 0:
            print("--Start--")
            print(" -->{0} self.query= {1}".format("startState", self.query))
        self.end_state = len(self.query)

        if self.query == 'twowords':
            print(self.query)
        return state
        # ### END CODE HERE ###

    def isEnd(self, state):
        # ### START CODE HERE ###
        if state == self.end_state:
            return True
        else:
            return False

        # ### END CODE HERE ###

    def succAndCost(self, state):
        # ### START CODE HERE ###


        if self.verbosity > 1:
            cost = sum([self.unigramCost(word) for word in state])
            print("-->{0} state= {1}".format("succAndCost", state),)

        def segment_successors(state, uni_cost_func):
            ri = state
            right_part = self.query[ri:]
            succ_states = []
            for i in range(1, len(right_part)+1):
                try:
                    new_word = right_part[:i]
                    tran_cost = uni_cost_func(new_word)
                    action = i
                    succ_states.append((action, ri+i, tran_cost))
                except Exception as e:
                    print(e)
            return succ_states
        next_states = segment_successors(state=state, uni_cost_func=self.unigramCost)
        return next_states
        # ### END CODE HERE ###


def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # ### START CODE HERE ###

    if True:
        print("totalCost: {0}".format(ucs.totalCost))
        print("actions: {0}".format(ucs.actions))
        print("numStatesExplored: {0}".format(ucs.numStatesExplored))

    query_cost = unigramCost(query)

    if ucs.totalCost > query_cost:
        print(" cost > query_cost {0}".format(query_cost))
        print(" res= {0}".format(query))
        return query

    if ucs.actions is None:
        print(" no actions".format(query_cost))
        print(" res= {0}".format(query))
        return query

    words = [query]
    for i, ai in enumerate(ucs.actions):
        try:
            last_word = words[-1]
            last_left = last_word[:ai]
            last_right = last_word[ai:]
            words = words[:-1] + [last_left, last_right]
        except Exception as e:
            print(e)

    res = ' '.join(words)
    res = res.rstrip()
    if True:
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
        self.verbosity = 0.5
        self.end = len(self.queryWords) - 1

        state = (-1, wordsegUtil.SENTENCE_BEGIN)
        if self.verbosity > 0:
            print("\n--Start--")
            print(" -->{0} queryWords {1}".format("start", self.queryWords),)
        return state
        # ### END CODE HERE ###

    def isEnd(self, state):
        # ### START CODE HERE ###

        if state[0] == self.end:
            return True
        else:
            return False
        # ### END CODE HERE ###

    def succAndCost(self, state):
        # ### START CODE HERE ###
        if self.verbosity > 1:
            print(" ->{0} state= {1}, ".format("succCost", state, ), )

        def replace_successors(state, bigramCost):
            if self.verbosity > 1:
                print("  -->{0} state= {1}, ".format("replace", state,),)

            wi, last_word = state
            if self.verbosity > 1:
                print(" queryWords {0}, wi {1}".format(self.queryWords, wi))
            try:
                next_frag = self.queryWords[wi+1]
            except Exception as e:
                import pdb
                pdb.set_trace()
            try:
                replace_words = self.possibleFills(next_frag)
            except Exception as e:
                pass
                replace_words = []

            succ_states = []
            # no replace
            next_cost = bigramCost(last_word, next_frag)
            action = (wi+1,(next_frag, next_frag))
            succ_states.append((action, (wi + 1, next_frag), next_cost))

            # using the fills
            try:
                replace_words = self.possibleFills(next_frag)
            except Exception as e:
                print("Exception calling possible fills with {0}".format(next_frag))
                print(e)
                replace_words = []

            for next_word in replace_words:
                next_cost = bigramCost(last_word, next_word)
                action = (wi+1, (next_frag, next_word),)
                succ_states.append((action, (wi+1, next_word), next_cost))
            return succ_states

        succ_states = replace_successors(state, self.bigramCost)
        if self.verbosity > 1:
            print("   next_states {0}, ".format(succ_states))
        return succ_states

        # ### END CODE HERE ###


def insertVowels(queryWords, bigramCost, possibleFills):
    # ### START CODE HERE ###
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))

    words = queryWords

    if ucs.actions is None or len(ucs.actions) == 0:
        return ' '.join(queryWords)

    if True:
        print("Done: {0} actions".format(len(ucs.actions)))

    for wi, (old_word, new_word) in ucs.actions:
        if words[wi] != old_word:
            msg = "error expecting {0} at pos {1} but found {2} in {3}".format(old_word, wi,
                                                                               words[wi], words)
            raise ValueError(msg)
        words[wi] = new_word
    res = ' '.join(words)
    if True:
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
        self.verbosity = 0
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
        if self.verbosity > 0:
            print("\n\n--Start 3--")
            print(" -->>{0} state= {1} init_cost: {2}  best_end: {3}".format("start",
                                                                             state,
                                                                             self.init_cost,
                                                                             self.best_end),)

        return state
        # ### END CODE HERE ###

    def isEnd(self, state):
        # ### START CODE HERE ###
        if state[0] == self.end_state:
            if self.verbosity > 0:
                print("  -->> {0} end-state= {1}".format("isEnd", state,), )
            res = True
        else:
            res = False
        return res
        # ### END CODE HERE ###

    def succAndCost(self, state):

        # ### START CODE HERE ###
        words = state

        if self.verbosity > 1:
            print("\n-->> '{0}' state= {1}, words: {2} ".format("succAndCost", state, words))
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
            if self.verbosity > 1:
                print("   -> {0} state= {1}, cur_cost: {2}".format("segSucc", state, cur_cost),)
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
            if self.verbosity > 1:
                print("   -> {0} state= {1}, cur_cost: {2}".format("replace", state, cur_cost),)
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
            if self.verbosity > 1:
                print("  succ_states: [{0}]".format(succ_states),)
            return succ_states

        segment_states = segment_successors(cur_words, apply_bigram_cost)
        replace_states = replace_successors(cur_words, apply_bigram_cost)

        next_states = replace_states + segment_states

        if cur_words[0] != self.query and cur_words[0] != self.end_state:
            if state_cost < self.best_end:
                self.best_end = state_cost
                if self.verbosity > 0:
                    print("** adding link to end state from {0} with cost {1} cur_words {2}".format(state, state_cost, cur_words),)
                next_states.append(( ('N', 0, 0), self.end_state, state_cost,))


        if self.verbosity > 0.9:
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
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))
    words = [query,]
    pairs = zip(words, words[1:])
    init_cost = sum([bigramCost(a, b) for a, b in pairs])
    if len(words) > 0:
        init_cost += bigramCost(wordsegUtil.SENTENCE_BEGIN, words[0])
    if False:
        print("Func: '{0}' query= {1}, init_cost: {2}".format("segmentAndInsert", query, init_cost),)
        print(" DONE: totalCost: {0}, actions: {1}".format(ucs.totalCost, ucs.actions))
        print("  NumStatesExplored: {0}".format(ucs.numStatesExplored,))

    if ucs.totalCost is None:
        return query

    #if ucs.totalCost >= init_cost:
    #    return query

    if ucs.actions is None or len(ucs.actions) == 0:
        return query

    for ai, (atype, wi, ainfo) in enumerate(ucs.actions):
        if True:
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
    if True > 0:
        print(" **answer: {0}, joined: {1}".format(words, res))
        print("**res: {0}, cost: {1}".format(res, ucs.totalCost))
        print("")
    return res
    # ### END CODE HERE ###

############################################################


if __name__ == '__main__':

    shell.main()
