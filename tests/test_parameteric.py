import sys
from pystl import *
import numpy as np
import time

# Build a contract
c1 = contract('c1')                                    # Create a contract c1
c1.add_nondeter_vars(['w'],  mean = [0], \
            cov = [[1**2]], dtypes=['GAUSSIAN']) # Set nondeterministic uncontrolled variables
c1.add_param_vars(['p', 'c'], bounds = [[0, 1], [-4, 4]])
# c1.add_deter_vars(['u'], bounds = [[-0.1, 0.1]])      # Set a controlled variable
c1.printInfo() # Print c
c1.set_assume('True') # Set/define the assumptions
# c1.set_guaran('P[p] (w + u <= c)') # Set/define the guarantees
c1.set_guaran('P[p] (w <= c)') # Set/define the guarantees
c1.checkSat()  # Saturate c
c1.printInfo() # Print c

start = time.time()
c1.find_opt_param({'p': -10, 'c': 1}, N=500)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))

# Build a contract
c2 = contract('c2')                                    # Create a contract c2
c2.add_nondeter_vars(['w'],  mean = ['mean'], \
            cov = [['sigma^2']], dtypes=['GAUSSIAN'])     # Set nondeterministic uncontrolled variables
c2.add_param_vars(['mean', 'sigma'], bounds = [[-0.1, 0.1], [0.01, 0.1]])
# c2.add_deter_vars(['u'], bounds = [[-0.1, 0.1]])      # Set a controlled variable
c2.printInfo() # Print c
c2.set_assume('True') # Set/define the assumptions
# c2.set_guaran('P[p] (w + u <= c)') # Set/define the guarantees
c2.set_guaran('P[0.99] (w <= 0)') # Set/define the guarantees
c2.checkSat()  # Saturate c
c2.printInfo() # Print c

start = time.time()
c2.find_opt_param({'mean': 1, 'sigma': -10}, N=400)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))