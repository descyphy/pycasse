import sys
from pystl import *
from pystl.parser import *
import numpy as np
import time

# Build a contract
c = contract('c')
[e1, e2] = c.set_deter_uncontrolled_vars(['e1', 'e2'], \
        # bounds = np.array([[0, 35], [0, 10**4]]))
        bounds = np.array([[-10**4, 10**4], [-10**4, 10**4]])) # Set a deterministic uncontrolled variable
[theta] = c.set_controlled_vars(['theta'], 
        # bounds = np.array([[-4, 4]]))
        bounds = np.array([[-10**4, 10**4]])) # Set a controlled variable
c.set_assume('True') # Set/define the assumptions
# c.set_guaran('((F[0,100] ((-0.01 <= e1) & (e1 <= 0.01))) & (0 <= e1) & (e1 <= 35) & (0 <= e2) & (e2 <= 35) & (e1 = e2))') # Set/define the guarantees
c.set_guaran('((!(F[0,100] ((-0.01 <= e1) & (e1 <= 0.01)))) & (0 <= e1) & (e1 <= 35) & (0 <= e2) & (e2 <= 35) & (e1 = e2))') # Set/define the guarantees
# c.set_guaran('(G[0,100] (((4 <= theta1) -> (theta2 = 4)) & ((theta1 <= -4) -> (theta2 = -4)) & (((-4 <= theta1) & (theta1 <= 4)) -> (theta2 = theta1))))') # Set/define the guaranteess
# c.set_guaran('((F[0,20] ((e1 <= 0.01) &/ (-0.01 <= e1))) & (G[0,20] (((4 <= theta1) -> (theta2 = 4)) & ((theta1 <= -4) -> (theta2 = -4)) & (((-4 <= theta1) & (theta1 <= 4)) -> (theta2 = theta1)))))') # Set/define the guarantees
c.checkSat()  # Saturate c
c.printInfo() # Print c

# Build a linear system dynamics
e = Vector([e1, e2])
theta = Vector([theta])
A_c = np.array([[0.9048, 0], [0.09516, 1]])
B_c = np.array([[-0.1903], [-0.009675]])
K_c = np.array([[-0.5635, -2.2316]])

# Build a MILP solver
solver = MILPSolver()
solver.add_contract(c)
solver.add_dynamic(Next(e) == A_c * e + B_c * theta)
solver.add_dynamic(theta == -K_c * e)
# solver.add_constraint(e1 == 20)
# solver.add_constraint(e2 == 20)
solver.add_constraint(c.guarantee)

# Solve the problem using MILP solver
start = time.time()
solved = solver.solve()
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
if solved:
    solver.print_solution()