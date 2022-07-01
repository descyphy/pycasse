import sys
from pystl import *
import numpy
import time

car = contract('car')
car.add_deter_vars(['xe', 've', 'xl', 'vl', 'ae', 'dm', 'vm'],
    bounds=[[0, 200], [0, 34], [0, 200], [0, 34], [-2.95, 1.99], [-100, 100], [-34, 34]])

car.add_param_vars(['p', 'c'], bounds=[[0.9, 1], [10, 40]]) # Design space for safety specification
# car.add_param_vars(['c1', 'c2'], bounds=[[0, 10], [0, 10]]) # Design space for recovery specification
# car.add_param_vars(['p', 'c'], bounds=[[0.9, 1], [-15, 0]]) # Design space for comfort specification

car.set_assume(
    '(xl - xe >= 30) & (ve - 5 <= vl) & (vl <= ve + 5)'
)

car.set_guaran(
    'G[0,20] (P[p] (xl - xe >= c))' # Safety specification
    # 'G[0,10] ((P[0.9] (ve >= vl + c1)) -> (F[0,10] (P[0.99] (ve + c2 <= vl))))' # Recovery specification
    # 'G[0,20] (P[p] (ae => c))' # Comfort specification
)

car.checkSat()
car.printInfo()
input()

K = 0.9
dT = 0.1

dynamics = {'x': ['xe', 've', 'xl', 'vl'], 
    'u': ['ae'], 
    'z': ['dm', 'vm'],
    'A': [[1, dT, 0, 0], [0, 1, 0, 0], [0, 0, 1, dT], [0, 0, 0, 1]],
    'B': [[0], [dT], [0], [0]], 
    'C': [[-1, 0, 1, 0], [0, -1, 0, 1]],
    'D': [[0, K]],
    'Q': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.5**2]],
    'R': [[0.5**2, 0], [0, 0.3**2]]
}

init_conditions = ['xe == 0', 've == 31', 'xl == 30', 'vl == 26']

start = time.time()
car.find_opt_param({'p': -100, 'c': -1}, N=200, dynamics=dynamics, init_conditions=init_conditions)
# car.find_opt_param({'p': -100, 'c': -1}, N=200, dynamics=dynamics)
# car.find_opt_param({'c1': -1, 'c2': -1}, N=100, dynamics=dynamics, init_conditions=init_conditions)
# car.find_opt_param({'c1': -10, 'c2': -1}, N=200, dynamics=dynamics)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
