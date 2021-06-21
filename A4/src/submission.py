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
    verbosity = 2
    if verbosity > 0:
      import datetime
      import sys

    if verbosity > 1:
      print(f"--getAction: <{datetime.datetime.now()}> gameState: \n{gameState}\n >")
      print(f"  depth {self.depth}")
      print("---")

    if self.depth == 0:
      return []

    legalMoves = gameState.getLegalActions()

    if gameState.isWin() or gameState.isLose():
      return []

    if len(legalMoves) == 0:
      return []

    def getBestIndices(func=max, xlist=None, choser=random.choice, verbosity=0):
      try:
        bestValue = func(xlist)
        bestIndices = [i for i in range(len(xlist)) if xlist[i] == bestValue]
        if len(bestIndices) == 0:
          print(f"  bestIndices is empty, bestValue: {bestValue}")
          chosenIndex = None
          res = []
        else:
          chosenIndex = choser(bestIndices)
      except Exception as e:
        bestValue = None
        bestIndices = []
        chosenIndex = None
        import sys
        (e1, e2, e3) = sys.exc_info()
        print(f" eval {e2}, etype: {e1}")
        print("**")

      return bestValue, bestIndices, chosenIndex

    def Vminmax(state, index, depth, verbosity=0):
      if verbosity > 2:
        print(f"  --Vminmax: depth: {depth} agentIndex: {index} <{datetime.datetime.now()}> ")

      if state.isLose() or state.isWin():
        return state.getScore()

      if depth == 0:
        return self.evaluationFunction(state)

      try:
        legalMoves = state.getLegalActions(agentIndex=index)
        if verbosity > 2:
          print(f" moves: {legalMoves}")
      except Exception as e:
        legalMoves = []
        import sys
        (e1, e2, e3) = sys.exc_info()
        print(f" eval {e2}, etype: {e1}")

      try:
        succStates = [state.generateSuccessor(index, m) for m in legalMoves]
      except Exception as e:
        succStates = []
        import sys
        (e1, e2, e3) = sys.exc_info()
        print(f" eval {e2}, etype: {e1}")

      if index < state.getNumAgents()-1:
        nextDepth = depth
      else:
        nextDepth = depth - 1

      nextIndex = (index + 1) % state.getNumAgents()
      try:
        if verbosity > 2:
          pass
        values = [Vminmax(ss, nextIndex, nextDepth, verbosity) for ss in succStates]
      except Exception as e:
        values = []
        import sys
        (e1, e2, e3) = sys.exc_info()
        print(f" eval {e2}, etype: {e1}")

      if index == 0:
        func = max
      else:
        func = min

      try:
        bestValue, bestIndices, chosenIndex = getBestIndices(func=func, xlist=values,
                                                           choser=min,
                                                           verbosity=verbosity)
      except Exception as e:
        bestValue = None
        bestIndices = []
        chosenIndex = None
        import sys
        (e1, e2, e3) = sys.exc_info()
        print(f" eval {e2}, etype: {e1}")
        print("**")

      res = values[chosenIndex]
      if verbosity > 2:
        print(f" values: {values}, func: {func}, bestValue: {bestValue} chosenIndex={chosenIndex} ")
        print(f"  choice action is {legalMoves[chosenIndex]} for agent {index}")
        print("")
      return res

    try:
      succStates = [gameState.generateSuccessor(0, m) for m in legalMoves]
    except Exception as e:
      import sys
      (e1, e2, e3) = sys.exc_info()
      print(f" eval {e2}, etype: {e1}")
      succStates = []

    try:
      moveValues = [Vminmax(s, 0, self.depth, verbosity=verbosity) for s in succStates]
    except Exception as e:
      import sys
      (e1, e2, e3) = sys.exc_info()
      print(f" eval {e2}, etype: {e1}")
      moveValues = []

    try:
      bestValue, bestIndices, chosenIndex = getBestIndices(func=max, xlist=moveValues,
                                                         choser=random.choice,
                                                         verbosity=verbosity)
    except Exception as e:
      bestValue = None
      bestIndices = []
      chosenIndex = None
      import sys
      (e1, e2, e3) = sys.exc_info()
      print(f" eval {e2}, etype: {e1}")
      print("**")

    if len(bestIndices) == 0:
      print(f"  bestIndices is empty, bestValue: {bestValue}")
      res = []
    else:
      if verbosity > 1:
        print(f"  chosenIndex: {chosenIndex} action: {legalMoves[chosenIndex]} len(legalMoves): {len(legalMoves)}")
      res = legalMoves[chosenIndex]
    # BEGIN_HIDE
    # END_HIDE

    return res

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
    pass
    # ### START CODE HERE ###
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
    pass
    # ### START CODE HERE ###
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
  # ### END CODE HERE ###

# Abbreviation
better = betterEvaluationFunction
