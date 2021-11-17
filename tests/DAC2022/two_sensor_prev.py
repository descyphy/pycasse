import sys
from pystl import *
import numpy as np
import time

# Build a contract c1
c1 = contract('c1') # Create a contract c
c1.add_nondeter_vars(['w1'],  mean = [0], \
            cov = [[0.02**2]]) # Set nondeterministic uncontrolled variables
c1.set_assume('True') # Set/define the assumptions
c1.set_guaran('P[0.95] (w1 <= 0.2)') # Set/define the guarantees
c1.checkSat()  # Saturate c
c1.printInfo() # Print c

# # Check compatibility, consistency, and feasibility of c1
# c1.checkCompat(print_sol=True) # Check compatibility of c1
# c1.checkConsis(print_sol=True) # Check consistency of c1
# c1.checkFeas(print_sol=True)   # Check feasibility of c1

# Build a contract c2
c2 = contract('c2') # Create a contract c2
c2.add_param_vars(['p', 'c', 'sigma2'], bounds = [[0.9, 1], [0.1, 0.5], [0.01, 0.3]])
c2.add_nondeter_vars(['w2'],  mean = [0], \
            cov = [['sigma2^2']]) # Set nondeterministic uncontrolled variables
c2.set_assume('True') # Set/define the assumptions
c2.set_guaran('P[p] (w2 <= c)') # Set/define the guarantees
c2.checkSat()  # Saturate c2
c2.printInfo() # Print c2

# Build a contract c
c = contract('c') # Create a contract c
c.add_param_vars(['sigma2'], bounds = [[0.01, 0.3]])
c.add_nondeter_vars(['w1', 'w2'],  mean = [0, 0], \
            cov = [[0.02**2, 0], [0, 'sigma2^2']], dtypes=['GAUSSIAN', 'GAUSSIAN']) # Set nondeterministic uncontrolled variables
c.set_assume('True') # Set/define the assumptions
c.set_guaran('P[0.99] (0.5*w1 + 0.5*w2 <= 0.1)') # Set/define the guarantees
c.checkSat()  # Saturate c
c.printInfo() # Print c

# Composition of c1 and c2
c12_comp = composition(c1, c2)        # Composition of c1 and c2
c12_comp.checkSat()                   # Saturate c12_comp
c12_comp.printInfo()                  # Print c12_comp

# Find optimal parameters
start = time.time()
c12_comp.find_opt_refine_param(c, {'p': 10, 'c': 1, 'sigma2': -1}, N=50)
# c.find_opt_param({'sigma2': -1}, N=200)
# c2.find_opt_param({'p': 10, 'c': 1, 'sigma2': -1}, N=200)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))