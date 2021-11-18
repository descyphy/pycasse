import sys
from pystl import *
import numpy
import time

cars = contract('cars')
cars.add_deter_vars(['v', 'vl', 'd', 'a', 'al', 'vn', 'vln']) ###### Bound these

cars.set_assume('G[0,10] ((al > -5) & (al < 5))')
cars.set_guaran('G[0,10] (d >= 0)')
# cars.set_guaran('G[0,10] (P[0.99] (d >= 0))')

solver = MILPSolver()
solver.add_contract(cars)

# solver.add_dynamics(
#     x = ['v', 'vl', 'd', 'a'], 
#     u = ['al'], 
#     w = ['vn', 'vln'],
#     A = [[1, 0, 0, 1], [0, 1, 0, 0], [-1, 1, 1, 0], [-2, 2, 0, 0]],
#     B = [[0], [1], [0], [0]],
#     C = [[0, 0], [0, 0], [0, 0], [2, -2]],
#     w_mean = [0, 0],
#     w_cov = [[0.1**2, 0],[0, 0.1**2]])

solver.add_dynamics(
    x = ['v', 'vl', 'd', 'a'], 
    u = ['al'],
    A = [[1, 0, 0, 1], [0, 1, 0, 0], [-1, 1, 1, 0], [-2, 2, 0, 0]],
    B = [[0], [1], [0], [0]])

cars.checkSat()
cars.printInfo()

# Add assumption constraints
solver.add_constraint(cars.assumption, name='b_a')
solver.add_constraint(cars.guarantee, name='b_g')

# Add initial conditions
solver.add_init_condition('v == 0')
solver.add_init_condition('vl == 10')
solver.add_init_condition('d == 20')
solver.add_init_condition('a == 0')

# Solve the problem using MILP solver
start = time.time()
solved = solver.solve()
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
if solved:
    solver.print_solution()
