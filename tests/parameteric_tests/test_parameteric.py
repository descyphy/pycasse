from pycasse import *
import time

c1 = contract('c1')                                       # Create a contract c1
c1.add_nondeter_vars(['w'],  mean = [0], \
            cov = [[2**2]], dtypes=['GAUSSIAN'])          # Set nondeterministic variables
c1.add_deter_vars(['u'], bounds = [[-0.5, 0.5]])          # Set deterministic variables
c1.add_param_vars(['p', 'c'], bounds = [[0, 1], [-4, 4]]) # Set parameteric variables
c1.set_assume('True')                                     # Set/define the assumptions
c1.set_guaran('P[p] (w + u <= c)')                        # Set/define the guarantees
c1.saturate()                                             # Saturate c
c1.printInfo()                                            # Print c

start = time.time()
c1.find_opt_param({'p': -10, 'c': 1}, N=200)              # Find the optimal parameters (p, c)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))


c2 = contract('c2')                                                       # Create a contract c2
c2.add_nondeter_vars(['w'],  mean = ['mean'], \
            cov = [['sigma^2']], dtypes=['GAUSSIAN'])                     # Set nondeterministic variables
c2.add_param_vars(['mean', 'sigma'], bounds = [[-0.1, 0.1], [0.01, 0.1]]) # Set parameteric variables
c2.set_assume('True')                                                     # Set/define the assumptions
c2.set_guaran('P[0.99] (w <= 0)')                                         # Set/define the guarantees
c2.saturate()                                                             # Saturate c
c2.printInfo()                                                            # Print c

start = time.time()
c2.find_opt_param({'mean': 1, 'sigma': -10}, N=200)                       # Find the optimal parameters (mean, sigma)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))