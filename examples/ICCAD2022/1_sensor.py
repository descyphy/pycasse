from pycasse import *
import time

# Build a contract c1
c1 = contract('c1') # Create a contract c1
c1.add_deter_vars(['x'], bounds=[[0,500]])
c1.add_param_vars(['p', 'sigma'], bounds = [[0.8, 1], [0.01, 0.3]])
c1.add_nondeter_vars(['w'],  mean = [0], \
            cov = [['sigma^2']]) # Set nondeterministic uncontrolled variables
# c1.set_assume('x<=200') # Set/define the assumptions
c1.set_guaran('P[p] (w <= 0.1)') # Set/define the guarantees
c1.checkSat()  # Saturate c2
c1.printInfo() # Print c2

# Find an optimal parameter for p
start = time.time()
c1.find_opt_param({'p': -100, 'sigma': 1}, N = 200)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))