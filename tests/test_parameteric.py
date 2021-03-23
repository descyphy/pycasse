import sys
sys.path.append("/home/kevin/Github/pystl")
from pystl import *
import numpy as np
import time

# Build a contract
c = contract('c') # Create a contract c
c.set_assume('True') # Set/define the assumptions
# c.set_guaran('(P[p] (<= w c))') # Set/define the guarantees
c.set_guaran('(P[p] (<= w+u c))') # Set/define the guarantees
c.set_nondeter_uncontrolled_vars(['w'],  mean = np.array([[0]]), \
                            cov = np.array([[1**2]]), dtype='GAUSSIAN') # Set nondeterministic uncontrolled variables
c.set_params(['p', 'c'], np.array([[0, 1], [-4, 4]]))
c.set_controlled_vars(['u'], bounds = np.array([0, 1])) # Set a controlled variable
c.saturate()  # Saturate c
c.printInfo() # Print c

c.find_opt_param('ad', N=100)
# c.checkCompat(print_sol=True) # Check compatibility of c
# c.checkConsis(print_sol=True) # Check consistency of c
# c.checkFeas(print_sol=True)   # Check feasibility of c

# # Build a MILP Solver
# MILPsolver = MILPSolver()
# MILPsolver.add_contract(c)
# # MILPsolver.add_dynamics(sys_dyn)

# # Solve the problem using MILP solver
# start = time.time()
# solved = MILPsolver.solve()
# end = time.time()
# print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
# if solved:
# 	for v in MILPsolver.MILP_convex_solver.getVars():
# 		if 'x' in v.varName or 'u' in v.varName:
# 			print('%s %g' % (v.varName, v.x))
