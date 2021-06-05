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
        self.verbosity = 0
        state = 0
        if self.verbosity > 0:
            print("--Start--")
            print(" -->{0} self.query= {1}".format("startState", self.query))
        self.end_state = len(self.query)


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

        def add_segments(state, uni_cost_func):
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
        next_states = add_segments(state=state, uni_cost_func=self.unigramCost)
        return next_states
        # ### END CODE HERE ###


def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # ### START CODE HERE ###
    verbosity = 0
    if verbosity > 1:
        print("totalCost: {0}".format(ucs.totalCost))
        print("actions: {0}".format(ucs.actions))
        print("numStatesExplored: {0}".format(ucs.numStatesExplored))

    query_cost = unigramCost(query)

    if ucs.totalCost > query_cost:
        if verbosity > 0:
            print(" cost > query_cost {0}".format(query_cost))
            print(" res= {0}".format(query))
        return query

    if ucs.actions is None:
        if verbosity > 0:
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
    if verbosity > 0:
        print("res: '{0}'".format(res))
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

        self.verbosity = 0
        self.end = len(self.queryWords)
        state = (0, wordsegUtil.SENTENCE_BEGIN)
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
        if self.verbosity > 0:
            print(" ->{0} state= {1}, ".format("succCost", state, ), )

        def add_replace(state, bigramCost):
            if self.verbosity > 1:
                print("  -->{0} state= {1}, ".format("replace", state,),)

            wi, last_word = state
            if self.verbosity > 1:
                print(" queryWords {0}, wi {1}".format(self.queryWords, wi))
            try:
                next_frag = self.queryWords[wi]
            except Exception as e:
                import pdb
                pdb.set_trace()

            succ_states = []
            # using the fills
            try:
                replace_words = self.possibleFills(next_frag)
            except Exception as e:
                print("Exception calling possible fills with {0}".format(next_frag))
                print(e)
                replace_words = [next]

            for next_word in replace_words:
                next_cost = bigramCost(last_word, next_word)
                action = (wi, (next_frag, next_word),)
                succ_states.append((action, (wi+1, next_word), next_cost))
            return succ_states

        succ_states = add_replace(state, self.bigramCost)
        if self.verbosity > 1:
            print("   next_states {0}, ".format(succ_states))
        return succ_states

        # ### END CODE HERE ###


def insertVowels(queryWords, bigramCost, possibleFills):
    # ### START CODE HERE ###
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))

    words = queryWords
    verbosity = 0
    if ucs.actions is None or len(ucs.actions) == 0:
        return ' '.join(queryWords)

    if True:
        if verbosity > 1:
            print("Done: {0} actions".format(len(ucs.actions)))

    for wi, (old_word, new_word) in ucs.actions:
        if words[wi] != old_word:
            msg = "error expecting {0} at pos {1} but found {2} in {3}".format(old_word, wi,
                                                                               words[wi], words)
            raise ValueError(msg)
        words[wi] = new_word
    res = ' '.join(words)
    if verbosity > 0:
        print("res: {0}".format(res))
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

        if self.query.startswith("r"):
            print("query= {0}".format(self.query))
        self.verbosity = 0
        self.end = len(self.query)
        state = (0, wordsegUtil.SENTENCE_BEGIN)

        if self.verbosity > 0:
            print("\n\n--Start 3--")
            print(" -->>{0}  queryWords= {1} ".format("startState", self.query))

        return state
        # ### END CODE HERE ###

    def isEnd(self, state):
        # ### START CODE HERE ###
        if state[0] == self.end:
            res = True
        else:
            res = False
        return res
        # ### END CODE HERE ###

    def succAndCost(self, state):

        # ### START CODE HERE ###

        if self.verbosity > 0:
            print("\n-->> '{0}' state= {1}, ".format("succAndCost", state))

        def add_segments(state, cost_func):
            ri, last_word = state
            right_part = self.query[ri:]
            succ_states = []
            for i in range(1, len(right_part)+1):
                try:
                    next_frag = right_part[:i]
                except Exception as e:
                    print(e)

                try:
                    replace_words = self.possibleFills(next_frag)
                except Exception as e:
                    print("Exception calling possible fills with {0}".format(next_frag))
                    print(e)
                    replace_words = [next_frag]
                for rword in replace_words:
                    tran_cost = cost_func(last_word, rword)
                    action = (i, (next_frag, rword))
                    succ_states.append((action, (ri+i, rword), tran_cost))

            return succ_states

        next_states = add_segments(state, self.bigramCost)
        if self.verbosity > 1:
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

    verbosity = 0
    if len(words) > 0:
        init_cost += bigramCost(wordsegUtil.SENTENCE_BEGIN, words[0])

    if verbosity > 1:
        print("Func: '{0}' query= {1}, init_cost: {2}".format("segmentAndInsert", query, init_cost),)
        print(" DONE: totalCost: {0}, actions: {1}".format(ucs.totalCost, ucs.actions))
        print("  NumStatesExplored: {0}".format(ucs.numStatesExplored,))

    if ucs.totalCost is None:
        if verbosity > 0:
            print(" totalCost is None, res: {0}".format(query))
        return query

    if ucs.actions is None or len(ucs.actions) == 0:
        if verbosity > 0:
            print(" actions is none or empty, res: {0}".format(query))
        return query

    for ai, action in enumerate(ucs.actions):
        if verbosity > 1:
            print("ai: {0}, action: {1}".format(ai, action))
        si, repl = action
        try:
            last_word = words[-1]
            last_left = last_word[:si]
            last_right = last_word[si:]
        except Exception as e:
            print(e)
        if last_left != repl[0]:
            raise ValueError(" expecing {0} but found {1}".format(repl[0], last_left))
        words = words[:-1] + [repl[1], last_right]

    res = ' '.join(words)
    res = res.rstrip()
    if verbosity > 0:
        print("res: {0} ".format(res,))
    return res
    # ### END CODE HERE ###

############################################################


if __name__ == '__main__':

    shell.main()
