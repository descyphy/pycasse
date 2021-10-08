import sys
from pystl import *
import numpy as np
import time

# Build a contract
c = contract('c')                                                # Create a contract c
c.set_nondeter_uncontrolled_vars(['w'],  mean = [0], \
            cov = [[1**2]], dtypes=['GAUSSIAN'])     # Set nondeterministic uncontrolled variables
c.set_params(['p', 'c'], bounds = [[0, 1], [-4, 4]])
# c.set_controlled_vars(['u'], bounds = [[-1, 1]])          # Set a controlled variable
c.set_assume('True') # Set/define the assumptions
# c.set_guaran('(P[p] (w+u <= c))') # Set/define the guarantees
c.set_guaran('(P[p] (0 <= w + c))') # Set/define the guarantees
c.checkSat()  # Saturate c
c.printInfo() # Print c

start = time.time()
c.find_opt_param([-100, 1], N=200)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))