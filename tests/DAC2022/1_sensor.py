import sys
from pystl import *
import numpy as np
import time

# Build a contract c1
c1 = contract('c1') # Create a contract c1
c1.add_deter_vars(['x'], bounds=[[0,500]])
# c1.add_deter_vars(['x', 'v'], bounds=[[0,500], [0, 34]])
c1.add_param_vars(['p', 'sigma'], bounds = [[0.8, 1], [0.01, 0.3]])
c1.add_nondeter_vars(['w'],  mean = [0], \
            cov = [['sigma^2']]) # Set nondeterministic uncontrolled variables
# c2.set_assume('x<=200') # Set/define the assumptions
c1.set_guaran('P[p] (w <= 0.1)') # Set/define the guarantees
c1.checkSat()  # Saturate c2
c1.printInfo() # Print c2

# # Build a contract c
# c = contract('c') # Create a contract c
# c.add_deter_vars(['x'], bounds=[[0,500]])
# c.add_param_vars(['sigma2'], bounds = [[0.01, 0.3]])
# c.add_nondeter_vars(['w1', 'w2'],  mean = [0, 0], \
#             cov = [[0.02**2, 0], [0, 'sigma2^2']], dtypes=['GAUSSIAN', 'GAUSSIAN']) # Set nondeterministic uncontrolled variables
# # c.set_assume('x<=200') # Set/define the assumptions
# c.set_guaran('P[0.99] (w1 + w2 <= 0.1)') # Set/define the guarantees
# c.checkSat()  # Saturate c
# c.printInfo() # Print c

# Find an optimal parameter for p
start = time.time()
c1.find_opt_param({'p': -100, 'sigma': 1}, N = 200)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))