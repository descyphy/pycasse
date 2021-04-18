from pystl import *
import numpy as np
import time

# Build a contract
c = contract('c') # Create a contract c
c.set_assume('True') # Set/define the assumptions
c.set_guaran('(P[p] (<= w c))') # Set/define the guarantees
# c.set_guaran('(P[p] (<= w+u c))') # Set/define the guarantees
c.set_nondeter_uncontrolled_vars(['w'],  mean = np.array([[0]]), \
                            cov = np.array([[1**2]]), dtype='GAUSSIAN') # Set nondeterministic uncontrolled variables
# c.set_params(['p', 'c'], np.array([[0, 1], [-4, 4]]))
c.set_params(['p', 'c'], np.array([[0.5, 0.7], [1, 2]]))
# c.set_controlled_vars(['u'], bounds = np.array([0, 1])) # Set a controlled variable
c.saturate()  # Saturate c
c.printInfo() # Print c

c.find_opt_param('ad')