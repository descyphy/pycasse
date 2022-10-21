from pycasse import *
import time

# Build a contract
c1 = contract('c1')                                   # Create a contract c1
c1.add_param_vars(['p', 'sigma'], bounds = [[0, 1], [0.05, 2]])
c1.add_nondeter_vars(['w'],  mean = [0], \
            cov = [['sigma^2']], dtypes=['GAUSSIAN']) # Set uncontrolled variables
c1.set_assume('True')                                 # Set/define the assumptions
c1.set_guaran('P[p] (w <= 1.5)')                      # Set/define the guarantees
c1.saturate()                                         # Saturate c1
c1.printInfo()                                        # Print c1

# Build a contract
c2 = contract('c2')                                   # Create a contract c2
c2.add_param_vars(['sigma'], bounds = [[0.05, 2]])
c2.add_nondeter_vars(['w'],  mean = [0], \
            cov = [['sigma^2']], dtypes=['GAUSSIAN']) # Set uncontrolled variables
c2.set_assume('True')                                 # Set/define the assumptions
c2.set_guaran('P[0.9] (w <= 1.5)')                    # Set/define the guarantees
c2.saturate()                                         # Saturate c2
c2.printInfo()                                        # Print c2

# Find an optimal set of parameters for refinement to hold
start = time.time()
c2.find_opt_refine_param(c1, {'p': -1, 'sigma': -1}, N=200)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))