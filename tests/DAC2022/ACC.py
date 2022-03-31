import sys
from pystl import *
import numpy
import time

car = contract('car')
car.add_deter_vars(['xe', 've', 'xl', 'vl', 'ae', 'd', 'v'],
    bounds=[[0, 500], [0, 34], [0, 500], [0, 34], [-2.95, 1.99], [-100, 100], [-34, 34]]) 

# car.add_param_vars(['p', 'vn_sigma', 'vln_sigma'],
#     bounds=[[0.9, 1], [0.0001, 0.3],[0.0001, 0.3]])

# car.add_param_vars(['p', 'dn_sigma'],
#     bounds=[[0, 1], [0.0001, 0.5]])

# car.add_param_vars(['p'], bounds=[[0.9, 1]])
car.add_param_vars(['c'], bounds=[[1, 5]])
# car.add_param_vars(['p', 'c'], bounds=[[0.9, 1], [1, 5]])

# car.add_param_vars(['vln_sigma'],
#     bounds=[[0.0001, 0.3]])

# car.add_nondeter_vars(['vn', 'vln', 'dn'],
#     mean=[0, 0, 0], cov=[['vn_sigma^2', 0, 0], 
#     [0, 'vln_sigma^2', 0], [0, 0, 'dn_sigma^2']])

# car.add_nondeter_vars(['vn', 'vln'],
#     mean=[0, 0], cov=[[0.01**2, 0], 
#     [0, 'vln_sigma^2']])

# car.add_nondeter_vars(['dn'],
#     mean=[0], cov=[['dn_sigma^2']])

# car.add_nondeter_vars(['dw'],
#     mean=[0], cov=[[0.1**2]])

car.set_assume(
    'True'
)

car.set_guaran(
    # '(G[0,10] (P[0.99] (d >= c))) & (G[0,10] (P[p] (a <= 0.9)))'
    # '(G[0,10] (P[0.99] (d >= c))) & (G[0,10] (P[0.9] (a <= 0.9)))'
    # '(G[0,10] (P[0.9](d >= c))) & (G[0,10]((P[0.01](d <= 0.1)) -> F[1,10] G[1,10](P[0.9999] (v <= vl)))'
    'G[0,10] (P[0.9] (d >= c))'
)
# cars.set_guaran(
# )

car.checkSat()
car.printInfo()

K = 1

dynamics = {'x': ['xe', 've', 'xl', 'vl'], 
    'u': ['ae'], 
    'z': ['d', 'v'],
    'A': [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
    'B': [[0], [1], [0], [0]], 
    'C': [[-1, 0, 1, 0], [0, -1, 0, 1]],
    'D': [[0, -K]],
    'Q': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1**2]],
    'R': [[2**2, 0], [0, 0.5**2]]
}

# TODO: dynamics addition function...

init_conditions = ['xe == 0', 'xl == 15', 've == 5', 'vl == 10', 'a == 0']

start = time.time()
# car.checkFeas(dynamics=dynamics, init_conditions=init_conditions, print_sol=True)
car.find_opt_param({'c': -1}, N=50, dynamics=dynamics, init_conditions=init_conditions)
# car.find_opt_param({'p': -1, 'c': -1}, N=50, dynamics=dynamics, init_conditions=init_conditions)
# car.find_opt_param({'p': -10, 'dn_sigma': 1}, N=50, dynamics=dynamics, init_conditions=init_conditions)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))

# #### How do we express the fact that the speed must be as close as possible to
# # the target speed?

# leading1 = contract('leading1')
# leading1.add_deter_vars(['vl', 'al'],
#     bounds = [[0,34], [-3.5, 3.5]])

# leading1.set_assume(
#     'True'
# )
# leading1.set_guaran(
#     '(G[0,10] (vl => 0)) & (G[0,10] (al => -2))'
# )

# leading2 = contract('leading2')
# leading2.add_deter_vars(['vl', 'al'],
#     bounds = [[0,34], [-3.5, 3.5]])

# leading2.set_assume(
#     'True'
# )
# leading2.set_guaran(
#     '(G[0,10] (vl => 0)) & (G[0,10] (al => -3))'
# )

# # Probabilistic version of the leading car.
# leadingp = contract('leadingp')
# leadingp.add_deter_vars(['vl', 'al'],
#     bounds = [[0,34], [-3.5, 3.5]])

# leadingp.set_assume(
#     'True'
# )

# leadingp.set_guaran(
#     '(G[0,10](v >= 0)) & \
#         (G[0,10] P[0.9](a >= -2))'
# )


# # Comfort contract
# comfort = contract('comfort')
# comfort.add_deter_vars(['a'], bounds = [-2.95,1.99])

# comfort.set_assume(
#     'True'
# )

# # Ref: Toward a Comfortable Driving Experience for a Self-Driving Shuttle Bus
# comfort.set_guaran(
#     '(G[0,10] (P[0.95] (a <= 0.9))) & \
#         (G[0,10] (P[0.95] (a >= -0.9)))'
# )

# cars.checkSat()
# cars.printInfo()

# # Add guarantee constraints
# solver.add_constraint(cars.guarantee, name='b_g')

# # Solve the problem using MILP solver
# start = time.time()
# solved = solver.solve()
# end = time.time()
# print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
# if solved:
#     solver.print_solution()
