import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ) )
from pystl import *
from pystl.parser import *
import numpy as np
import time

# Build a contract
c = contract('c')                               # Create a contract c
c.add_deter_vars(['s', 'v', 'a'], 
    bounds = [[-100, 2000], [-5, 10], [-1, 1]]) # Set a deterministic variables
c.add_nondeter_vars(['w'], \
        mean = [0], cov = [[1**2]])     
c.set_assume('True')                            # Set/define the assumptions
# c.set_guaran('F[0,10] (P[0.9] (s => 30))')      # Set/define the guarantees
c.set_guaran('F[0,3] (P[0.9] (s => 30))')      # Set/define the guarantees
c.checkSat()                                    # Saturate c
c.printInfo()                                   # Print c

# Build a linear system dynamics
solver = MILPSolver()
solver.add_contract(c)

# Build a linear system dynamics
solver.add_dynamics(x = ['s', 'v'], u = ['a'], w = ['w'], A = [[1, 1], [0, 1]], B = [[0], [1]], C = [[0], [1]])

# Add initial conditions
solver.add_init_condition('s == 0')
solver.add_init_condition('v == 0')

# Add guarantee constraints
solver.add_constraint(c.guarantee, name='b_g')

# Solve the problem using MILP solver
start = time.time()
solved = solver.solve()
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
if solved:
    solver.print_solution()
