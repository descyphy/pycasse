import sys
from pystl import *
import numpy
import time

## Initialize constants
K = 0.3
dT = 0.1
d_safe = 10
tau = 1.6
N = 200
INIT = False
if INIT:
    # init_conditions = ['xe == 0', 've == 30', 'xl == 30', 'vl == 25']
    init_conditions = ['xe == 0', 've == 30', 'xl == 58', 'vl == 25']

## Create dynamics
dynamics = {'x': ['xe', 've', 'xl', 'vl'], 
    'u': ['ae'], 
    'z': ['dm', 'vrm', 'vem'],
    'A': [[1, dT, 0, 0], [0, 1, 0, 0], [0, 0, 1, dT], [0, 0, 0, 1]],
    'B': [[0], [dT], [0], [0]], 
    'C': [[-1, 0, 1, 0], [0, -1, 0, 1], [0, 0, 1, 0]],
    'D': [[K, K, -tau*K]],
    'E': [[-d_safe*K]],
    'Q': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, (dT*1)**2]],
    'R': [[1**2, 0, 0],[0, 1**2, 0], [0, 0, 0.5**2]]
}

## ACC Safety
# Create acc safety contract
acc_safety = contract('acc_safety')
acc_safety.add_deter_vars(['xe', 've', 'xl', 'vl', 'ae', 'dm', 'vrm', 'vem'],
    bounds=[[0, 70], [0, 30], [0, 70], [0, 30], [-3, 2], [-100, 100], [-30, 30], [0, 30]])

acc_safety.add_param_vars(['c1', 'c2'], bounds=[[0, 20], [0, 20]]) # Design space for acc safety contract

acc_safety.set_assume(
    'xl - xe >= {} + {}*ve'.format(d_safe, tau) # Safety assumptions
)

acc_safety.set_guaran(
    'G[0,20] ((P[0.9] (xl - xe <= c1)) -> (F[0,10](G[0,10](P[0.9] (ve <= c2)))))' # Safety guarantee
)

acc_safety.checkSat()
acc_safety.printInfo()
input()

# Find parameters for acc safety contract
# {'p': -10, 'c': -1}  -> {'p': 0.93125, 'c': 9.375}
# {'p': -100, 'c': -1} -> {'p': 0.98125, 'c': 8.75}
start = time.time()
if INIT:
    opt_dict = acc_safety.find_opt_param({'c1': -1, 'c2': 1}, N=N, dynamics=dynamics, init_conditions=init_conditions)
else:
    opt_dict = acc_safety.find_opt_param({'c1': -1, 'c2': 1}, N=N, dynamics=dynamics)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
c_safe = float(opt_dict['c'])
input()

# ## ACC Recovery
# # Create acc recovery contract 
# acc_recovery = contract('acc_recovery')
# acc_recovery.add_deter_vars(['xe', 've', 'xl', 'vl', 'ae', 'dm', 'vrm', 'vem'],
#     bounds=[[0, 70], [0, 30], [0, 70], [0, 30], [-3, 2], [-100, 100], [-30, 30], [0, 30]])

# acc_recovery.add_param_vars(['p', 'c'], bounds=[[0.9, 1], [c_safe, 20]]) # Design space for safety specification

# acc_recovery.set_assume(
#     # '(xl - xe <= {} + {}*ve) & (xl - xe >= {} + {}*ve)'.format(d_safe, tau, opt_dict['c'], tau) # Recovery assumption
#     '(xl - xe <= {} + {}*ve) & (xl - xe >= {} + {}*ve)'.format(d_safe, tau, c_safe, tau) # Recovery assumption
# )

# acc_recovery.set_guaran(
#     'F[0,20] (P[p] (xl - xe >= c + {}*ve))'.format(tau) # Recovery guarantee
# )

# acc_recovery.checkSat()
# acc_recovery.printInfo()
# input()

# # Find parameters for acc recovery contract
# # {'p': -10, 'c': -1}  -> {'p': 0.95, 'c': 10.703125}
# # {'p': -100, 'c': -1} -> {'p': 0.98125, 'c': 9.453125}
# start = time.time()
# if INIT:
#     opt_dict = acc_recovery.find_opt_param({'p': -100, 'c': -1}, N=N, dynamics=dynamics, init_conditions=init_conditions)
# else:
#     opt_dict = acc_recovery.find_opt_param({'p': -100, 'c': -1}, N=N, dynamics=dynamics)
# end = time.time()
# print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
# input()

# ## ACC Recovery2
# # Create acc recovery contract 
# acc_recovery2 = contract('acc_recovery2')
# acc_recovery2.add_deter_vars(['xe', 've', 'xl', 'vl', 'ae', 'dm', 'vrm', 'vem'],
#     bounds=[[0, 70], [0, 30], [0, 70], [0, 30], [-3, 2], [-100, 100], [-30, 30], [0, 30]])

# acc_recovery2.add_param_vars(['p', 'c'], bounds=[[0.9, 1], [c_safe, 20]]) # Design space for safety specification

# acc_recovery2.set_assume(
#     '(xl - xe <= {} + {}*ve) & (xl - xe >= {})'.format(c_safe, tau, d_safe) # Recovery assumption
# )

# acc_recovery2.set_guaran(
#     'F[0,20] (P[p] (xl - xe >= c + {}*ve))'.format(tau) # Recovery guarantee
# )

# acc_recovery2.checkSat()
# acc_recovery2.printInfo()
# input()

# # Find parameters for acc recovery contract
# # {'p': -10, 'c': -1}  -> 
# # {'p': -100, 'c': -1} -> 
# start = time.time()
# if INIT:
#     opt_dict = acc_recovery2.find_opt_param({'p': -100, 'c': -1}, N=N, dynamics=dynamics, init_conditions=init_conditions)
# else:
#     opt_dict = acc_recovery2.find_opt_param({'p': -100, 'c': -1}, N=N, dynamics=dynamics)
# end = time.time()
# print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
# # input()

# ## ACC Comfort
# # Create acc contract for comfort
# acc_comfort = contract('acc_comfort')
# acc_comfort.add_deter_vars(['xe', 've', 'xl', 'vl', 'ae', 'dm', 'vrm', 'vem'],
#     bounds=[[0, 70], [0, 30], [0, 70], [0, 30], [-3, 2], [-100, 100], [-30, 30], [0, 30]])

# acc_comfort.add_param_vars(['p', 'c'], bounds=[[0.9, 1], [-50, 0]]) # Design space for safety specification

# acc_comfort.set_assume(
#     'xl - xe >= {}+ {}*ve'.format(d_safe, tau) # Comfort assumption
# )

# acc_comfort.set_guaran(
#     'G[0,20] (P[p] (c <= ae))' # Comfort guarantee
# )

# acc_comfort.checkSat()
# acc_comfort.printInfo()
# input()

# # Find parameters for acc comfort contract: {'p': 0.90625, 'c': -31.25}
# start = time.time()
# if INIT:
#     opt_dict = acc_comfort.find_opt_param({'p': -10, 'c': -1}, N=N, dynamics=dynamics, init_conditions=init_conditions)
# else:
#     opt_dict = acc_comfort.find_opt_param({'p': -10, 'c': -1}, N=N, dynamics=dynamics)
# end = time.time()
# print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))