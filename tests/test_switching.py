import sys
from pystl import *
from pystl.parser import *
import numpy as np
import time

# Build a contract
c = contract('c')
[vd, vl, vf, e1, e2, e3, xr] = c.set_deter_uncontrolled_vars(['vd', 'vl', 'vf', 'e1', 'e2', 'e3', 'xr'],
        bounds = np.array([[0, 35], [0, 10**4], [-10**4, 10**4], [-10**4, 10**4], [-10**4, 10**4], [-10**4, 10**4], [-10**4, 10**4]])) # Set a deterministic uncontrolled variable
[theta] = c.set_controlled_vars(['theta'], 
        bounds = np.array([[-10**4, 10**4]])) # Set a controlled variable
c.set_assume('True') # Set/define the assumptions
c.set_guaran('(F[0,100] ((-0.01 <= e1) & (e1 <= 0.01) & (-0.01 <= e2) & (e2 <= 0.01)))') # Set/define the guarantees
# c.set_guaran('(!(F[0,100] ((-0.01 <= e1) & (e1 <= 0.01))))') # Set/define the guarantees
c.checkSat()  # Saturate c
c.printInfo() # Print c

# Build a linear system dynamics
e = Vector([e1, e2, e3])
theta = Vector([theta])
x = Vector([xr])
v = Vector([vl, vf])
A_c = np.array([[0.9048, 0, 0], [0, 1, 0], [0.09516, 0, 1]])
B_c = np.array([[-0.1903], [0], [-0.009675]])
K_c = np.array([[-0.5635, 0, -2.2316]])
A_f = np.array([[0.9048, 0, 0], [0, 1, 0], [0.09516, 0.1, 1]])
B_f = np.array([[-0.1903], [-0.2], [-0.01967]])
K_f = np.array([[-18.2544, 15.3248, -11.158]])

# Build a MILP solver
solver = MILPSolver()

# Check property
solver.add_contract(c)
solver.add_constraint(c.guarantee)

# Dynamics
solver.add_switching_dynamic([[Next(e) == A_c * e + B_c * theta, theta == -K_c * e], [Next(e) == A_f * e + B_f * theta, theta == -K_f * e]], switching_time = 70)

# Conditions that has to always hold
solver.add_dynamic(np.array([[1,0,0]]) * e == np.array([[1,0]]) * v - np.array([[0,1]]) * v) # e1 = vl - vf
solver.add_dynamic(np.array([[1,0]]) * v == 30) # vl = 30

# Initial conditions
solver.add_constraint(vf == 10)
# solver.add_constraint(e1 == 20)
# solver.add_constraint(e2 == 0)

# Solve the problem using MILP solver
start = time.time()
solved = solver.solve()
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
if solved:
    solver.print_solution()