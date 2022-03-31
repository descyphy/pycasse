import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ) )
from pystl import *
from pystl.parser import *
import numpy as np
import time

# Build a contract
c = contract('c')                               # Create a contract c
c.add_deter_vars(['s', 'v', 'a'], 
    bounds = [[-100, 2000], [-5, 10], [-1, 2]]) # Set a deterministic variables
c.add_nondeter_vars(['w'], 
    mean = [0], cov = [[0.2**2]])     
c.add_param_vars(['p', 'c'], 
    bounds = [[0, 1], [20, 60]])
c.set_assume('G[0,9] (a == 1)')                 # Set/define the assumptions
c.set_guaran('F[0,10] (P[p] (s => c))')         # Set/define the guarantees
c.checkSat()                                    # Saturate c
c.printInfo()                                   # Print c

# Dynamics
dynamics = {'x': ['s', 'v'], 
    'u': ['a'], 
    'w': ['w'],
    'A': [[1, 1], [0, 1]],
    'B': [[0], [1]], 
    'C': [[0], [1]]
}

# Initial conditions
init_conditions = ['s == 0', 'v == 0']

# Find an optimal parameter for p
start = time.time()
c.find_opt_param({'p': -100, 'c': -1}, N = 200, dynamics = dynamics, init_conditions = init_conditions)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
