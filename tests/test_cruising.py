import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ) )

from pystl import *
from pystl.parser import *
import numpy as np
import time

# Build a contract
c = contract('c')
[vd, vf, e1, e2] = c.set_deter_uncontrolled_vars(['vd', 'vf', 'e1', 'e2'], \
        bounds = np.array([[0, 35], [0, 10**4], [-10**4, 10**4], [-10**4, 10**4]])) # Set a deterministic uncontrolled variable
[theta] = c.set_controlled_vars(['theta'], 
        bounds = np.array([[-10**4, 10**4]])) # Set a controlled variable
c.set_assume('True') # Set/define the assumptions
# c.set_guaran('(F[0,100] (G[0,50] ((-0.01 <= e1) & (e1 <= 0.01))))') # Set/define the guarantees
c.set_guaran('(!(F[0,100] (G[0,50] ((-0.01 <= e1) & (e1 <= 0.01)))))') # Set/define the guarantees
c.checkSat()  # Saturate c
c.printInfo() # Print c

# Build a linear system dynamics
e = Vector([e1, e2])
theta = Vector([theta])
v = Vector([vf, vd])
A_c = np.array([[0.9048, 0], [0.09516, 1]])
B_c = np.array([[-0.1903], [-0.009675]])
K_c = np.array([[-0.5635, -2.2316]])

# Build a MILP solver
solver = MILPSolver()

# Check property
solver.add_contract(c)
solver.add_hard_constraint(c.guarantee)

# Dynamics
solver.add_dynamic(Next(e) == A_c * e + B_c * theta)

# Conditions that has to always hold
solver.add_dynamic(theta == -K_c * e)
solver.add_dynamic(np.array([[1,0]]) * v == np.array([[0,1]]) * v - np.array([[1,0]]) * e)
solver.add_dynamic(Next(np.array([[0,1]]) * v) == np.array([[0,1]]) * v)
# solver.add_dynamic(np.array([[0,1]]) * v == 30)

# Initial conditions
# solver.add_hard_constraint(vf == 10)
# solver.add_hard_constraint(e1 == 20)
solver.add_hard_constraint(e2 == 0)

# Solve the problem using MILP solver
solver.preprocess()
start = time.time()
solved = solver.solve()
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
if solved:
    solver.print_solution()
