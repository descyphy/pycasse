import sys
from pystl import *
from pystl.parser import *
import numpy as np
import time

# Build a contract
c = contract('c')
[sl, sf, vl, vf] = c.set_deter_uncontrolled_vars(['sl', 'sf', 'vl', 'vf'], \
        bounds = np.array([[0, 500], [0, 500], [0, 35], [0, 35]]))                    # Set a deterministic uncontrolled variable
[al, af] = c.set_controlled_vars(['al', 'af'], bounds = np.array([[-4, 4], [-4, 4]])) # Set a controlled variable
c.set_assume('True') # Set/define the assumptions
c.set_guaran('(!(G[0,10] (sf <= sl)))') # Set/define the guarantees
c.saturate()  # Saturate c
c.printInfo() # Print c

# Build a linear system dynamics
xl = Dynamics([sl, vl])
ul = Dynamics([al])
xf = Dynamics([sf, vf])
uf = Dynamics([af])
A = np.array([[1, 1], [0, 1]])
B = np.array([[0], [1]])
# A_c = np.array([[0.9048, 0], [0.09516, 1]])
# B_c = np.array([[-0.1903], [-0.009675]])
# K_c = np.array([[-0.5635, -2.2316]])

solver = MILPSolver()
solver.add_contract(c)
solver.add_dynamic(Next(xl) == xl * A + ul * B)
solver.add_dynamic(Next(xf) == xf * A + uf * B)
# solver.add_dynamic(Next(e) == e * A + theta * B)
# solver.add_dynamic(Next(theta) == e * -K)
solver.add_constraint(sl == 4.5)
solver.add_constraint(sf == 0)
solver.add_constraint(vl == 0)
solver.add_constraint(vf == 0)
solver.add_constraint(c.guarantee)

# Solve the problem using MILP solver
start = time.time()
solved = solver.solve()
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
if solved:
    solver.print_solution()