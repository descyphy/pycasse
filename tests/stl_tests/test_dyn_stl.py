import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ) )
from pycasse import *
from pycasse.parser import *
import time

# Build a contract
c = contract('c')                                               # Create a contract c
c.add_deter_vars(['s', 'v', 'a'], 
    bounds = [[-100, 1000], [-5, 10], [-1, 1]])                 # Set deterministic variables
c.set_assume('True')                                            # Set/define the assumptions
# c.set_guaran('F[0,50] (s >= 445)')                              # Set/define the guarantees
# c.set_guaran('F[0,100] (s => 945)')                             # Set/define the guarantees
c.set_guaran('G[0,10] ((F[0,5] (s => 3)) & (F[0,5] (s <= 0)))') # Set/define the guarantees
c.checkSat()                                                    # Saturate c
c.printInfo()                                                   # Print c

# Build a linear system dynamics
solver = MILPSolver()
solver.add_contract(c)

# Build a linear system dynamics
solver.add_dynamics(x = ['s', 'v'], u = ['a'], A = [[1, 1], [0, 1]], B = [[0], [1]])

# Add initial conditions
solver.add_init_condition('s == 0')
solver.add_init_condition('v == 0')

# Add guarantee constraints
solver.add_constraint(c.guarantee, name='b_g')

# Solve the problem using MILP solver
start = time.time()
solved = solver.solve()
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
if solved:
    solver.print_solution()
