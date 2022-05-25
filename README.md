# Hill Climbing Algorithm for Eight Queens Problem
## AIM

To develop a code to solve eight queens problem using the hill-climbing algorithm.

## THEORY:
 In a chess board 8 queens should be placed in such a way that no one should be able to attack each other.This can be done using hill climbing algorithm.
 Hill climbing algorithm is a local search algorithm which continuously moves in the direction of increasing elevation/value to find the peak of the mountain or best solution to the problem. It terminates when it reaches a peak value where no neighbor has a higher value.Itis a technique which is used for optimizing the mathematical problems.

## DESIGN STEPS: 
### STEP 1:
Import the necessary libraries

### STEP 2:
Define the Intial State and calculate the objective function for that given state

### STEP 3:
Make a decision whether to change the state with a smaller objective function value, or stay in the current state.

### STEP 4:
Repeat the process until the total number of attacks, or the Objective function, is zero.

### STEP 5:
Display the necessary states and the time taken.

## PROGRAM:
### Main program:
```python
%matplotlib inline
import matplotlib.pyplot as plt
import random
import math
import sys
from collections import defaultdict, deque, Counter
from itertools import combinations
from IPython.display import display
from ipynb.fs.full.plot import plot_NQueens
import time
class Problem(object):
    """The abstract class for a formal problem. A new domain subclasses this,
    overriding `actions` and `results`, and perhaps other methods.
    The default heuristic is 0 and the default action cost is 1 for all states.
    When yiou create an instance of a subclass, specify `initial`, and `goal` states 
    (or give an `is_goal` method) and perhaps other keyword args for the subclass."""

    def __init__(self, initial=None, goal=None, **kwds): 
        self.__dict__.update(initial=initial, goal=goal, **kwds) 
        
    def actions(self, state):        
        raise NotImplementedError
    def result(self, state, action): 
        raise NotImplementedError
    def is_goal(self, state):        
        return state == self.goal
    def action_cost(self, s, a, s1): 
        return 1
    
    def __str__(self):
        return '{0}({1}, {2})'.format(
            type(self).__name__, self.initial, self.goal)
class Node:
    "A Node in a search tree."
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

    def __str__(self): 
        return '<{0}>'.format(self.state)
    def __len__(self): 
        return 0 if self.parent is None else (1 + len(self.parent))
    def __lt__(self, other): 
        return self.path_cost < other.path_cost
failure = Node('failure', path_cost=math.inf) # Indicates an algorithm couldn't find a solution.
cutoff  = Node('cutoff',  path_cost=math.inf) # Indicates iterative deepening search was cut off.
def expand(problem, state):
    return problem.actions(state)
class NQueensProblem(Problem):

    def __init__(self, N):
        super().__init__(initial=tuple(random.randint(0,N-1) for _ in tuple(range(N))))
        self.N = N

    def actions(self, state):
        """ finds the nearest neighbors"""
        neighbors = []
        for i in range(self.N):
            for j in range(self.N):
                if j == state[i]:
                    continue
                s1 = list(state)
                s1[i]=j
                new_state = tuple(s1)
                yield Node(state=new_state)

    def result(self, state, row):
        """Place the next queen at the given row."""
        col = state.index(-1)
        new = list(state[:])
        new[col] = row
        return tuple(new)

    def conflicted(self, state, row, col):
        """Would placing a queen at (row, col) conflict with anything?"""
        return any(self.conflict(row, col, state[c], c)
                   for c in range(col))

    def conflict(self, row1, col1, row2, col2):
        """Would putting two queens in (row1, col1) and (row2, col2) conflict?"""
        return (row1 == row2 or  # same row
                col1 == col2 or  # same column
                row1 - col1 == row2 - col2 or  # same \ diagonal
                row1 + col1 == row2 + col2)  # same / diagonal

    def goal_test(self, state):
        return not any(self.conflicted(state, state[col], col)
                       for col in range(len(state)))

    def h(self, node):
        """Return number of conflicting queens for a given node"""
        num_conflicts = 0
        for (r1,c1) in enumerate(node.state):
            for (r2,c2) in enumerate(node.state):
                if(r1,c1) != (r2,c2):
                    num_conflicts += self.conflict(r1,c1,r2,c2)
        return num_conflicts
def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items
def argmin_random_tie(seq, key):
    """Return an element with highest fn(seq[i]) score; break ties at random."""
    return min(shuffled(seq), key=key)
def hill_climbing(problem,iterations = 10000):
    # as this is a stochastic algorithm, we will set a cap on the number of iterations        
    current = Node(problem.initial)
    i=1
    while i < iterations:
        neighbors = expand(problem,current.state)
        if not neighbors:
            break
        neighbor = argmin_random_tie(neighbors,key=lambda node:problem.h(node))
        if problem.h(neighbor) <= problem.h(current):
            """Note that it is based on neggative path cost method"""
            current.state = neighbor.state
            if problem.goal_test(current.state)==True:
                print("Goal test succeeded at iteration {0}".format(i))
                return current
        i += 1        
    return current 
nq1=NQueensProblem(8)
plot_NQueens(nq1.initial)
n1 = Node(state=nq1.initial)
num_conflicts = nq1.h(n1)
print("Initial Conflicts = {0}".format(num_conflicts))
sol1=hill_climbing(nq1,iterations=20000)
print(sol1.state)
num_conflicts = nq1.h(sol1)
print("Final Conflicts = {0}".format(num_conflicts))
plot_NQueens(list(sol1.state))
n=[2,4,8,16,32,64]
s=[]
for i in range(6):
    start_time = time.time()
    nq1=NQueensProblem(n[i])
    n1 = Node(state=nq1.initial)
    num_conflicts = nq1.h(n1)
    sol1=hill_climbing(nq1,iterations=20000)
    time_taken=time.time() - start_time
    print("--- {0} seconds ---".format(time_taken))
    s.append(time_taken)
print(s)
plt.plot(n,s)
plt.show()
```
### plot.py:
```python
import time
from collections import defaultdict
from inspect import getsource

import ipywidgets as widgets
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from IPython.display import HTML
from IPython.display import display
from PIL import Image
from matplotlib import lines

# Function to plot NQueensCSP in csp.py and NQueensProblem in search.py
def plot_NQueens(solution):
    n = len(solution)
    board = np.array([2 * int((i + j) % 2) for j in range(n) for i in range(n)]).reshape((n, n))
    im = Image.open('images/queen_s.png')
    height = im.size[1]
    im = np.array(im).astype(np.float) / 255
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.set_title('{} Queens'.format(n))
    plt.imshow(board, cmap='binary', interpolation='nearest')
    # NQueensCSP gives a solution as a dictionary
    if isinstance(solution, dict):
        for (k, v) in solution.items():
            newax = fig.add_axes([0.064 + (k * 0.112), 0.062 + ((7 - v) * 0.112), 0.1, 0.1], zorder=1)
            newax.imshow(im)
            newax.axis('off')
    # NQueensProblem gives a solution as a list
    elif isinstance(solution, list):
        for (k, v) in enumerate(solution):
            newax = fig.add_axes([0.064 + (k * 0.112), 0.062 + ((7 - v) * 0.112), 0.1, 0.1], zorder=1)
            newax.imshow(im)
            newax.axis('off')
    elif isinstance(solution, tuple):
        for (k, v) in enumerate(solution):
            newax = fig.add_axes([0.064 + (k * 0.112), 0.062 + ((7 - v) * 0.112), 0.1, 0.1], zorder=1)
            newax.imshow(im)
            newax.axis('off')    
    fig.tight_layout()
    plt.show()
```

## OUTPUT:
![Screenshot (211)](https://user-images.githubusercontent.com/75234807/170296465-fbf509f7-31b3-453e-9dc6-53c186b23ea5.png)
![Screenshot (212)](https://user-images.githubusercontent.com/75234807/170296548-d93d9c19-a891-4c26-9b5f-a0f17ebd7b14.png)

## Time Complexity Plot
![Screenshot (213)](https://user-images.githubusercontent.com/75234807/170296587-d0b8164a-2235-488c-9e36-60fb8e8eff4c.png)

## RESULT:
Hence, a code to solve eight queens problem using the hill-climbing algorithm has been implemented.
