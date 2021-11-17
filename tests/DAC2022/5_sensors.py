import sys
from pystl import *
import numpy as np
import time

# Build a contract c1
c1 = contract('c1') # Create a contract c
c1.add_deter_vars(['x'], bounds=[[0,500]])
c1.add_nondeter_vars(['w1'],  mean = [0], \
            cov = [[0.02**2]]) # Set nondeterministic uncontrolled variables
# c1.set_assume('x<=300') # Set/define the assumptions
c1.set_guaran('P[0.95] (w1 <= 0.2)') # Set/define the guarantees
c1.checkSat()  # Saturate c
c1.printInfo() # Print c

# Build a contract c2
c2 = contract('c2') # Create a contract c2
c2.add_deter_vars(['x'], bounds=[[0,500]])
c2.add_param_vars(['p', 'sigma2'], bounds = [[0.8, 1], [0.01, 0.3]])
c2.add_nondeter_vars(['w2'],  mean = [0], \
            cov = [['sigma2^2']]) # Set nondeterministic uncontrolled variables
# c2.set_assume('x<=200') # Set/define the assumptions
c2.set_guaran('P[p] (w2 <= 0.1)') # Set/define the guarantees
c2.checkSat()  # Saturate c2
c2.printInfo() # Print c2

# Build a contract c3
c3 = contract('c3') # Create a contract c3
c3.add_deter_vars(['x'], bounds=[[0,500]])
c3.add_nondeter_vars(['w3'],  mean = [0], \
            cov = [[0.02**2]]) # Set nondeterministic uncontrolled variables
# c3.set_assume('x<=300') # Set/define the assumptions
c3.set_guaran('P[0.9] (w3 <= 0.1)') # Set/define the guarantees
c3.checkSat()  # Saturate c3
c3.printInfo() # Print c3

# Build a contract c4
c4 = contract('c4') # Create a contract c4
c4.add_deter_vars(['x'], bounds=[[0,500]])
c4.add_nondeter_vars(['w4'],  mean = [0], \
            cov = [[0.02**2]]) # Set nondeterministic uncontrolled variables
# c4.set_assume('x<=300') # Set/define the assumptions
c4.set_guaran('P[0.9] (w4 <= 0.1)') # Set/define the guarantees
c4.checkSat()  # Saturate c4
c4.printInfo() # Print c4

# Build a contract c5
c5 = contract('c5') # Create a contract c5
c5.add_deter_vars(['x'], bounds=[[0,500]])
c5.add_nondeter_vars(['w5'],  mean = [0], \
            cov = [[0.02**2]]) # Set nondeterministic uncontrolled variables
# c5.set_assume('x<=300') # Set/define the assumptions
c5.set_guaran('P[0.9] (w5 <= 0.1)') # Set/define the guarantees
c5.checkSat()  # Saturate c5
c5.printInfo() # Print c5

# Build a contract c
c = contract('c') # Create a contract c
c.add_deter_vars(['x'], bounds=[[0,500]])
c.add_param_vars(['sigma2'], bounds = [[0.01, 0.3]])
c.add_nondeter_vars(['w1', 'w2', 'w3', 'w4', 'w5'],  mean = [0, 0, 0, 0, 0], \
            cov = [[0.02**2, 0, 0, 0, 0], [0, 'sigma2^2', 0, 0, 0], [0, 0, 0.02**2, 0, 0], [0, 0, 0, 0.02**2, 0], [0, 0, 0, 0, 0.02**2]]) # Set nondeterministic uncontrolled variables
# c.set_assume('x<=200') # Set/define the assumptions
c.set_guaran('P[0.99] (w1 + w2 + w3 + w4 + w5 <= 0.25)') # Set/define the guarantees
c.checkSat()  # Saturate c
c.printInfo() # Print c

# Composition
c12_comp = composition(c1, c2)        # Composition of c1 and c2
c123_comp = composition(c12_comp, c3) # Composition of c12_comp and c3
c1234_comp = composition(c123_comp, c4) # Composition of c123_comp and c4
c12345_comp = composition(c1234_comp, c5) # Composition of c1234_comp and c5
c12345_comp.checkSat()                  # Saturate c12345_comp
c12345_comp.printInfo()                 # Print c12345_comp

# Find optimal parameters
start = time.time()
c12345_comp.find_opt_refine_param(c, {'p': -1, 'sigma2': -10}, N=200)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))