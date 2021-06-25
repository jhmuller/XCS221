from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
# BEGIN_HIDE
# END_HIDE

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    # BEGIN_HIDE
    # END_HIDE

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # BEGIN_HIDE
    # END_HIDE
    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

# noinspection PyPep8Naming
class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # ### START CODE HERE ###

    def getBestIndices(func=max, xlist=None, choser=random.choice):
      try:
        bestValue = func(xlist)
        bestIndices = [i for i in range(len(xlist)) if xlist[i] == bestValue]
        chosenIndex = choser(bestIndices)
        return bestValue, bestIndices, chosenIndex
      except Exception as e:
        print(str(e))
        raise RuntimeError(e)

    def Vminmax(state, index, depth, moveHist, verbosity=0):
      if moveHist is None:
        moveHist = []
      if verbosity > 1:
        pass
        print(f"  --Vminmax: agent {index }  depth: {depth}")
        # print(f"  {moveHist}")

      if state.isLose() or state.isWin():
        return state.getScore()

      if depth == 0:
        #print(f"   score {evalScore} index: {index} depth: {depth}")
        evalScore = self.evaluationFunction(state)
        return evalScore

      nextDepth = depth
      if index == state.getNumAgents()-1:
        nextDepth -= 1
      nextIndex = (index + 1) % state.getNumAgents()

      try:
        legalMoves = state.getLegalActions(agentIndex=index)
        succStates = [state.generateSuccessor(index, m) for m in legalMoves]
        values = [Vminmax(ss, nextIndex, nextDepth, verbosity) for ss in succStates]
        func = max if index == 0 else min
        bestValue, bestIndices, chosenIndex = getBestIndices(func=func, xlist=values,
                                                             choser=min)
        res = values[chosenIndex]
        return res
      except Exception as e:
        print(str(e))
        raise RuntimeError(e)

    #### getAction ####
    try:
      verbosity = 1
      if hasattr(self, "callNumber"):
        self.callNumber += 1
      else:
        self.callNumber = 0

      if verbosity > 1:
        print(f"--getAction: depth: {self.depth} callNumber: {self.callNumber}")
        if verbosity > 2:
          print(f"gameState: \n{gameState}\n >")

      if gameState.isWin() or gameState.isLose():
        return []

      legalMoves = gameState.getLegalActions()
      succStates = [gameState.generateSuccessor(0, m) for m in legalMoves]
      nextIndex = (self.index + 1) % gameState.getNumAgents()
      moveValues = [Vminmax(s, nextIndex, self.depth, moveHist=[], verbosity=verbosity) for s in succStates]

      bestValue, bestIndices, chosenIndex = getBestIndices(func=max, xlist=moveValues,
                                                           choser=random.choice)
      if len(bestIndices) == 0:
        raise RuntimeError("  bestIndices is empty")
      res = legalMoves[chosenIndex]

      if verbosity > 0.5:
        if self.callNumber == 0:
          print(f"  *** bestValue: {bestValue} callNumber {self.callNumber} chosenIndex: {chosenIndex} action: {res}")
          #print(f"   len(legalMoves): {len(legalMoves)}")
          print("---")
      return res
    except Exception as e:
      print(str(e))
      raise RuntimeError(e)
    # ### END CODE HERE ###

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # ### START CODE HERE ###

    def getCallStack(depths=None, verbosity=0):
      import sys
      import inspect
      if depths is None:
        depths = [1]
      if not isinstance(depths,list):
        depths = [depths]
      res = []
      for i, d in enumerate(depths):
        try:
          fobj = sys._getframe(d)
          fname = fobj.f_code.co_name
          if verbosity > 2:
            print(f"{d}  {fname} {fobj.f_locals}")
            print("---")
        except Exception as e:
          name = None
        res.append((d, fobj, fname))
      return res

    def getBestIndices(func=max, xlist=None, choser=random.choice):
      try:
        bestValue = func(xlist)
        bestIndices = [i for i in range(len(xlist)) if xlist[i] == bestValue]

        if len(bestIndices) == 0:
          raise RuntimeError(f"  bestIndices is empty, bestValue: {bestValue}")

        chosenIndex = choser(bestIndices)
        return bestValue, bestIndices, chosenIndex
      except Exception as e:
        print(str(e))
        raise RuntimeError(str(e))

    def VminmaxPrune(state, index, depth,
                     lower_bound=float("-inf"),
                     upper_bound=float("inf"),
                     verbosity=0):
      if verbosity > 2:
        print(f"  --VminmaxPrune: depth: {depth} agentIndex: {index} > ")
        print(f"  lower_bound: {lower_bound} upper_bound: {upper_bound}")
        print(f"  state: \n{state}")

      if state.isLose() or state.isWin():
        return state.getScore()

      if depth == 0:
        return self.evaluationFunction(state)

      legalMoves = state.getLegalActions(agentIndex=index)
      succStates = [state.generateSuccessor(index, m) for m in legalMoves]

      nextDepth = depth
      if index == state.getNumAgents() - 1:
        nextDepth -= 1

      nextIndex = (index + 1) % state.getNumAgents()

      try:
        # visit descendents in order of scores
        stateScores = [self.evaluationFunction(s) for s in succStates]
        reverse = index == 0
        scoreTups = sorted([(y, x) for x, y in list(enumerate(stateScores))], reverse=reverse)

        valueTups = []
        for score, si in scoreTups:
          ss = succStates[si]
          value = VminmaxPrune(ss, nextIndex, nextDepth,
                               lower_bound=lower_bound,
                               upper_bound=upper_bound,
                               verbosity=verbosity)
          if index == 0:
            if value > lower_bound:
              lower_bound = value
          else:
            if value < upper_bound:
              upper_bound = value

          if lower_bound >= upper_bound:
            if verbosity > 2:
              print(f" {upper_bound} < {lower_bound} No overlap of upper and lower bounds, breaking")
            continue
          valueTups.append((value, si))

        if len(valueTups) == 0:
          res = float("inf")
          if index == 0:
            res = float("-inf")
          return res

        func = max if index == 0 else min
        values = [e1 for e1, e2 in valueTups]
        bestValue, bestIndices, chosenIndex = getBestIndices(func=func, xlist=values,
                                                             choser=min)
        res = values[chosenIndex]
        if verbosity > 2:
          print(f" values: {values}, func: {func}, bestValue: {bestValue} chosenIndex={chosenIndex} ")
          print(f"  choice action is {legalMoves[chosenIndex]} for agent {index}")
          print("----")
      except Exception as e:
         print("in VminimaxPrune")
         raise RuntimeError(e)
      return res

    ## getState ##
    if hasattr(self, "callNumber"):
      self.callNumber += 1
    else:
      if not hasattr(self, "verbosity"):
        self.verbosity = 0
        print(f" {self.verbosity}")
      self.callNumber = 0
      if self.verbosity > 1:
        print(f"  numAgents: {gameState.getNumAgents()}")

    if self.verbosity > 1:
      print(f"--getAction: depth: {self.depth} callNumber: {self.callNumber} ")
      if self.verbosity > 1:
        print(f"gameState: \n{gameState}\n >")

    try:
      if self.depth == 0:
        raise ValueError("depth is 0")

      if gameState.isWin() or gameState.isLose():
        return []
      legalMoves = gameState.getLegalActions()

      if len(legalMoves) == 0:
        raise ValueError("no legal moves")

      succStates = [gameState.generateSuccessor(0, m) for m in legalMoves]
      nextIndex = (self.index + 1) % gameState.getNumAgents()
      moveValues = [VminmaxPrune(ss, nextIndex, self.depth, lower_bound=float("-inf"),
                                 upper_bound=float("inf"), verbosity=self.verbosity)
                    for ss in succStates]

      bestValue, bestIndices, chosenIndex = getBestIndices(func=max, xlist=moveValues,
                                                             choser=random.choice)
      if len(bestIndices) == 0:
        raise RuntimeError("len(bestIndices) == 0")

      res = legalMoves[chosenIndex]
      if self.verbosity > 1:
        print(f"  *** bestValue: {bestValue} chosenIndex: {chosenIndex} action: {legalMoves[chosenIndex]}")
        print(f"   len(legalMoves): {len(legalMoves)}")
        print("---")

      return res
    except Exception as e:
      print("my error")
      raise RuntimeError(e)
    # ### END CODE HERE ###

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    # ### START CODE HERE ###
    def getBestIndices(func=max, xlist=None, choser=random.choice):
      try:
        bestValue = func(xlist)
        bestIndices = [i for i in range(len(xlist)) if xlist[i] == bestValue]

        if len(bestIndices) == 0:
          raise RuntimeError(f"  bestIndices is empty, bestValue: {bestValue}")

        chosenIndex = choser(bestIndices)
        return bestValue, bestIndices, chosenIndex
      except Exception as e:
        print(str(e))
        raise RuntimeError(str(e))

    def VminmaxExp(state, index, depth,
                     lower_bound=float("-inf"),
                     upper_bound=float("inf"),
                     verbosity=0):
      if verbosity > 2:
        print(f"  --VminmaxPrune: depth: {depth} agentIndex: {index} > ")
        print(f"  lower_bound: {lower_bound} upper_bound: {upper_bound}")
        print(f"  state: \n{state}")

      if state.isLose() or state.isWin():
        return state.getScore()

      if depth == 0:
        return self.evaluationFunction(state)

      legalMoves = state.getLegalActions(agentIndex=index)
      succStates = [state.generateSuccessor(index, m) for m in legalMoves]

      nextDepth = depth
      if index == state.getNumAgents() - 1:
        nextDepth -= 1
      nextIndex = (index + 1) % state.getNumAgents()
      try:
        # visit descendents in order of scores
        stateScores = [self.evaluationFunction(s) for s in succStates]
        reverse = index == 0
        scoreTups = sorted([(y, x) for x, y in list(enumerate(stateScores))], reverse=reverse)
        valueTups = []
        for score, si in scoreTups:
          ss = succStates[si]
          value = VminmaxExp(ss, nextIndex, nextDepth,
                               lower_bound=lower_bound,
                               upper_bound=upper_bound,
                               verbosity=verbosity)
          valueTups.append((value, si))

        if len(valueTups) == 0:
          res = float("inf")
          if index == 0:
            res = float("-inf")
          return res

        values = [e1 for e1, e2 in valueTups]

        if index == 0:
          bestValue, bestIndices, chosenIndex = getBestIndices(func=max, xlist=values,
                                                               choser=min)
          res = values[chosenIndex]
        else:
          res = float(sum(values)) / len(values)

        if verbosity > 2:
          print(f" res = {res} values: {values} ")
          print("----")
      except Exception as e:
         print("in VminimaxPrune")
         raise RuntimeError(e)
      return res

    ## getState ##
    if hasattr(self, "callNumber"):
      self.callNumber += 1
    else:
      if not hasattr(self, "verbosity"):
        self.verbosity = 0
        print(f" {self.verbosity}")
      self.callNumber = 0
      if self.verbosity > 1:
        print(f"  numAgents: {gameState.getNumAgents()}")

    if self.verbosity > 1:
      print(f"--getAction: depth: {self.depth} callNumber: {self.callNumber} ")
      if self.verbosity > 1:
        print(f"gameState: \n{gameState}\n >")

    try:
      if self.depth == 0:
        raise ValueError("depth is 0")

      if gameState.isWin() or gameState.isLose():
        return []
      legalMoves = gameState.getLegalActions()

      if len(legalMoves) == 0:
        raise ValueError("no legal moves")

      succStates = [gameState.generateSuccessor(0, m) for m in legalMoves]
      nextIndex = (self.index + 1) % gameState.getNumAgents()
      moveValues = [VminmaxExp(ss, nextIndex, self.depth, lower_bound=float("-inf"),
                                 upper_bound=float("inf"), verbosity=self.verbosity)
                    for ss in succStates]

      bestValue, bestIndices, chosenIndex = getBestIndices(func=max, xlist=moveValues,
                                                             choser=random.choice)
      if len(bestIndices) == 0:
        raise RuntimeError("len(bestIndices) == 0")

      res = legalMoves[chosenIndex]
      if self.verbosity > 1:
        print(f"  *** bestValue: {bestValue} chosenIndex: {chosenIndex} action: {legalMoves[chosenIndex]}")
        print(f"   len(legalMoves): {len(legalMoves)}")
        print("---")

      return res
    except Exception as e:
      print("my error")
      raise RuntimeError(e)
    # ### END CODE HERE ###

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
    Your extreme, unstoppable evaluation function (problem 4).

    DESCRIPTION: <write something here so we know what you did>
  """
  pass
  # ### START CODE HERE ###
  verbosity = 0

  def futurePacPositions(state, maxTimeSteps=0):
    positions = set([currentGameState.getPacmanPosition()])
    curStates = [state]
    for ti in range(maxTimeSteps):
      moves = [s.getLegalActions(agentIndex=0) for s in curStates]
      curStates = [state.generateSuccessor(agentIndex=0, action=move) for move in moves]
      newPos = set([ns.getPacmanPosition() for ns in curStates])
      positions.append(newPos)
    return positions

  class Node(object):
    def __init__(self, id, pos):
      self.id = id
      self.pos = pos


  class Graph(object):
    def __init__(self):
      from collections import defaultdict
      self.nodes = dict()
      self.edges = set()

    def isNode(self, node):
      try:
        if id in self.nodes:
          return True
        return False
      except Exception as e:
        import sys
        print(sys.exc_info())
        raise RuntimeError(str(e))

    def makeGraph(self, walls):
      self.wallData = walls.data
      self.cellsData = [[not c for c in row] for row in walls.data]
      for ri, row in enumerate(self.cellsData):
        for ci, col in enumerate(row):
          try:
            if col:
              self.addNode((ri, ci))
              if ri > 0:
                if self.cellsData[ri-1][ci]:
                  self.addEdge((ri, ci), (ri-1, ci))
              if ci > 0:
                if self.cellsData[ri][ci - 1]:
                  self.addEdge((ri, ci), (ri, ci - 1))
          except Exception as e:
            import sys
            print(f" {ri} {ci} self.cellsData[ri][ci]")
            print(sys.exc_info())
            raise RuntimeError(str(e))
      return None

    def addNode(self, pos):
      id = 0
      if len(self.nodes) > 0:
        id = max([v  for k,v in self.nodes.items()]) + 1
      try:
        self.nodes[pos] = id
      except Exception as e:
        import sys
        print(sys.exc_info())
        raise RuntimeError(str(e))

    def addEdge(self, pos1, pos2):
      try:
        id1 = self.nodes[pos1]
        id2 = self.nodes[pos2]
        self.edges.add(( (id1, pos1), (id2, pos2) ))
        #self.edges[id1].add(id2)
        #self.edges[id2].add(id2)
      except Exception as e:
        import sys
        print(sys.exc_info())
        raise RuntimeError(str(e))


    def allPairsShortestPaths(self):
      nodes = sorted(list(self.nodes))
      print(f" {len(self.nodes)} nodes")
      dist = []
      for i in range(len(self.nodes)):
        dist.append([float("inf") for n in range(len(self.nodes))])

      # distance for all edges
      for i, e in  enumerate(self.edges):
        n1 = e[0]
        n2 = e[1]
        dist[n1[0]][n2[0]] = 1
      numNodes = len(self.nodes)
      for k in range(numNodes):
        for i in range(numNodes):
          for j in range(numNodes):
            if dist[i][j] > dist[i][k] + dist[k][j]:
              dist[i][j] = dist[i][k] + dist[k][j]
      self.dist = dist
      print('here')
      return None


  def distance(stae, p1, p2):
    pass

  def foodScore(state, timeSteps):
    pass

  def dangerScore(state):
    pass

  def winScore(state):
    pass

  print(f"{currentGameState}")

  ghostPositions = currentGameState.getGhostPositions()
  pacmanPosition = currentGameState.getPacmanPosition()
  numFood = currentGameState.getNumFood()
  numAgents = currentGameState.getNumAgents()
  capsules = currentGameState.getCapsules()
  walls = currentGameState.getWalls()

  gameGraph = Graph()
  gameGraph.makeGraph(walls)
  gameGraph.allPairsShortestPaths()
  oldFood = currentGameState.getFood()
  print(oldFood)
  print("here")
  if False:


    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

  # ### END CODE HERE ###

# Abbreviation
better = betterEvaluationFunction

def run(layname, pac, ghosts, nGames = 1, name = 'games', catchExceptions=True):
  """
  Runs a few games and outputs their statistics.
  """
  import pacman, time, layout, textDisplay
  starttime = time.time()
  lay = layout.getLayout(layname, 3)
  disp = textDisplay.NullGraphics()

  print(('*** Running %s on' % name, layname,'%d time(s).' % nGames))
  games = pacman.runGames(lay, pac, ghosts, disp, nGames, False, catchExceptions=catchExceptions)
  print(('*** Finished running %s on' % name, layname,'after %d seconds.' % (time.time() - starttime)))

  stats = {'time': time.time() - starttime, 'wins': [g.state.isWin() for g in games].count(True), 'games': games, 'scores': [g.state.getScore() for g in games], 'timeouts': [g.agentTimeout for g in games].count(True)}
  print(('*** Won %d out of %d games. Average score: %f ***' % (stats['wins'], len(games), sum(stats['scores']) * 1.0 / len(games))))

  return stats
