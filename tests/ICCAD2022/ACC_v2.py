import sys
from pystl import *
import numpy
import time

## Initialize constants
K = 0.5
dT = 0.5
d_safe = 10
tau = 1.6
N = 200
INIT = False
if INIT:
    init_conditions = ['xe == 0', 've == 30', 'xl == 58', 'vl == 25']

## Create dynamics
dynamics = {'x': ['xe', 've', 'xl', 'vl'], 
    'u': ['ae'], 
    'z': ['dm', 'vrm', 'vem'],
    'A': [[1, dT, 0, 0], [0, 1, 0, 0], [0, 0, 1, dT], [0, 0, 0, 1]],
    'B': [[0], [dT], [0], [0]], 
    'C': [[-1, 0, 1, 0], [0, -1, 0, 1], [0, 1, 0, 0]],
    'D': [[K, K, -tau*K]],
    'E': [[-d_safe*K]],
    'Q': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, (dT*1)**2]],
    'R': [[1**2, 0, 0],[0, 1**2, 0], [0, 0, 0.5**2]]
}

# ## ACC Safety
# # w/ init  -> {'p': 0.9875, 'c': 42.1875} 453.0341 s
# # w/o init -> {'p': 0.9875, 'c': 2.5} 439.1022 s
# # Create acc safety contract
# acc_safety = contract('acc_safety')
# acc_safety.add_deter_vars(['xe', 've', 'xl', 'vl', 'ae', 'dm', 'vrm', 'vem'],
#     bounds=[[0, 70], [0, 30], [0, 70], [0, 30], [-3, 2], [-100, 100], [-30, 30], [0, 30]])

# acc_safety.add_param_vars(['p', 'c'], bounds=[[0.9, 1], [0, 10]]) # Design space for acc safety contract

# acc_safety.set_assume(
#     # 'xl - xe >= {} + {}*ve'.format(d_safe, tau) # Safety assumptions
#     '(xl - xe >= {}+ {}*ve) & (ve - 5 <= vl) & (vl <= ve + 5)'.format(d_safe, tau) # Safety assumptions
#     # 'True'
# )

# acc_safety.set_guaran(
#     'G[0,20] (P[p] (xl - xe >= c))' # Safety guarantee
# )

# acc_safety.checkSat()
# acc_safety.printInfo()
# # input()

# # Find parameters for acc safety contract
# start = time.time()
# if INIT:
#     opt_dict = acc_safety.find_opt_param({'p': -100, 'c': -1}, N=N, dynamics=dynamics, init_conditions=init_conditions)
# else:
#     opt_dict = acc_safety.find_opt_param({'p': -100, 'c': -1}, N=N, dynamics=dynamics)
# end = time.time()
# print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
# # c_safe = float(opt_dict['c'])
# # input()

# ## ACC Recovery
# # Create acc recovery contract 
# acc_recovery = contract('acc_recovery')
# acc_recovery.add_deter_vars(['xe', 've', 'xl', 'vl', 'ae', 'dm', 'vrm', 'vem'],
#     bounds=[[0, 70], [0, 30], [0, 70], [0, 30], [-3, 2], [-100, 100], [-30, 30], [0, 30]])

# acc_recovery.add_param_vars(['p', 'c'], bounds=[[0.8, 1], [0, 20]]) # Design space for safety specification

# acc_recovery.set_assume(
#     # '(xl - xe <= {} + {}*ve) & (xl - xe >= {} + {}*ve) & (ve - 5 <= vl) & (vl <= ve + 5)'.format(d_safe, tau, c_safe, tau) # Recovery assumption
#     # '(xl - xe <= {} + {}*ve) & (xl - xe >= {})'.format(c_safe, tau, d_safe) # Recovery assumption
#     '(xl - xe <= {}) & (ve - 5 <= vl) & (vl <= ve + 5)'.format(8.75) # Recovery assumption
# )

# acc_recovery.set_guaran(
#     # 'F[0,20] (P[p] (xl - xe >= c + {}*ve))'.format(tau) # Recovery guarantee
#     'F[0,20] (P[p] (xl - xe >= c))' # Recovery guarantee
# )

# acc_recovery.checkSat()
# acc_recovery.printInfo()
# input()

# # Find parameters for acc recovery contract
# # {'p': -10, 'c': -1}  -> {'p': 0.95, 'c': 10.703125}
# # {'p': -100, 'c': -1} -> {'p': 0.98125, 'c': 9.453125} {'p': 0.9875, 'c': 13.75}
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

# acc_recovery2.add_param_vars(['p', 'c'], bounds=[[0.9, 1], [0, 20]]) # Design space for safety specification

# acc_recovery2.set_assume(
#     '(xl - xe <= {} + {}*ve) & (xl - xe >= {}) & (ve - 5 <= vl) & (vl <= ve + 5)'.format(c_safe, tau, d_safe) # Recovery assumption
# )

# acc_recovery2.set_guaran(
#     'F[0,20] (P[p] (xl - xe >= c + {}*ve))'.format(tau) # Recovery guarantee
# )

# acc_recovery2.checkSat()
# acc_recovery2.printInfo()
# input()

# # Find parameters for acc recovery2 contract
# # {'p': -10, 'c': -1}  -> 
# # {'p': -100, 'c': -1} -> {'p': 0.99375, 'c': 13.75}
# start = time.time()
# if INIT:
#     opt_dict = acc_recovery2.find_opt_param({'p': -100, 'c': -1}, N=N, dynamics=dynamics, init_conditions=init_conditions)
# else:
#     opt_dict = acc_recovery2.find_opt_param({'p': -100, 'c': -1}, N=N, dynamics=dynamics)
# end = time.time()
# print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
# input()

## ACC Comfort 1
# Create acc contract for comfort 
# w/o init: {'p': 0.9875, 'c': -9.375}, 448.2608 s
# w/  init: {'p': 0.99375, 'c': -5.0},  431.5720 s
acc_comfort1 = contract('acc_comfort1')
acc_comfort1.add_deter_vars(['xe', 've', 'xl', 'vl', 'ae', 'dm', 'vrm', 'vem'],
    bounds=[[0, 70], [0, 30], [0, 70], [0, 30], [-3, 2], [-100, 100], [-30, 30], [0, 30]])

acc_comfort1.add_param_vars(['p', 'c'], bounds=[[0.9, 1], [-30, 0]]) # Design space for safety specification

acc_comfort1.set_assume(
    # 'xl - xe >= {} + {}*ve'.format(d_safe, tau) # Comfort assumption
    '(xl - xe >= {}+ {}*ve) & (ve - 5 <= vl) & (vl <= ve + 5)'.format(d_safe, tau) # Safety assumptions
    # 'True' # Safety assumptions
)

acc_comfort1.set_guaran(
    'G[0,3] (P[p] (c <= ae))' # Comfort guarantee
    # 'G[0,20] (P[p] (c <= ae))' # Comfort guarantee
)

acc_comfort1.checkSat()
acc_comfort1.printInfo()
# input()

# Find parameters for acc comfort contract: 
start = time.time()
if INIT:
    opt_dict = acc_comfort1.find_opt_param({'p': -100, 'c': -1}, N=N, dynamics=dynamics, init_conditions=init_conditions)
else:
    opt_dict = acc_comfort1.find_opt_param({'p': -100, 'c': -1}, N=N, dynamics=dynamics)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
# input()

# ## ACC Comfort 2
# # Create acc contract for comfort
# # w/o init: ,  s
# acc_comfort2 = contract('acc_comfort2')
# acc_comfort2.add_deter_vars(['xe', 've', 'xl', 'vl', 'ae', 'dm', 'vrm', 'vem'],
#     bounds=[[0, 70], [0, 30], [0, 70], [0, 30], [-3, 2], [-100, 100], [-30, 30], [0, 30]])

# acc_comfort2.add_param_vars(['p', 'c'], bounds=[[0.9, 1], [0, 80]]) # Design space for safety specification

# acc_comfort2.set_assume(
#     'xl - xe >= {} + {}*ve'.format(d_safe, tau) # Comfort assumption
#     # '(xl - xe >= {}+ {}*ve) & (ve - 5 <= vl) & (vl <= ve + 5)'.format(d_safe, tau) # Safety assumptions
#     # 'True' # Safety assumptions
# )

# acc_comfort2.set_guaran(
#     'G[0,20] (P[p] (ae <= c))' # Comfort guarantee
# )

# acc_comfort2.checkSat()
# acc_comfort2.printInfo()
# # input()

# # Find parameters for acc comfort contract: 
# # w/ init: {'p': 0.99375, 'c': -5.0} 471.6650
# start = time.time()
# if INIT:
#     opt_dict = acc_comfort2.find_opt_param({'p': -100, 'c': -1}, N=N, dynamics=dynamics, init_conditions=init_conditions)
# else:
#     opt_dict = acc_comfort2.find_opt_param({'p': -100, 'c': -1}, N=N, dynamics=dynamics)
# end = time.time()
# print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))