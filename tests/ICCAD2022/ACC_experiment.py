from pystl import *
import time

DEBUG = False

## Initialize constants
K = 0.5
dT = 0.5
d_safe = 10
tau = 1.6
sigma_al = 0.5
N = 200
INIT = False
if INIT:
    init_conditions = ['xe == 0', 've == 30', 'xl == 58', 'vl == 25']
    # init_conditions = ['xe == 0', 've == 0', 'xl == 70', 'vl == 0']

## Create dynamics
dynamics = {'x': ['xe', 've', 'xl', 'vl'], 
    'u': ['ae'], 
    'z': ['dm', 'vrm', 'vem'],
    'A': [[1, dT, 0, 0], [0, 1, 0, 0], [0, 0, 1, dT], [0, 0, 0, 1]],
    'B': [[0], [dT], [0], [0]], 
    'C': [[-1, 0, 1, 0], [0, -1, 0, 1], [0, 1, 0, 0]],
    'D': [[K, K, -tau*K]],
    'E': [[-d_safe*K]],
    'Q': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, (dT*sigma_al)**2]],
    'R': [[1**2, 0, 0],[0, 1**2, 0], [0, 0, 0.5**2]]
}

## ACC Safety
# Create acc safety contract
acc_safety = contract('acc_safety')
acc_safety.add_deter_vars(['xe', 've', 'xl', 'vl', 'ae', 'dm', 'vrm', 'vem'],
    bounds=[[0, 50], [0, 30], [0, 50], [0, 30], [-3, 2], [-100, 100], [-30, 30], [0, 30]])

acc_safety.add_param_vars(['p', 'c'], bounds=[[0.9, 1], [0, 10]]) # Design space for acc safety contract

acc_safety.set_assume(
    '(xl - xe >= {} + {}*ve) & (ve - 5 <= vl) & (vl <= ve + 5)'.format(d_safe, tau) # Safety assumptions
)

acc_safety.set_guaran(
    'G[0,20] (P[p] (xl - xe >= c))' # Safety guarantee
)

acc_safety.checkSat()
acc_safety.printInfo()
# input()

# Find parameters for acc safety contract
start = time.time()
if INIT:
    opt_dict = acc_safety.find_opt_param({'p': -100, 'c': -1}, N=N, dynamics=dynamics, init_conditions=init_conditions, debug=DEBUG)
else:
    opt_dict = acc_safety.find_opt_param({'p': -100, 'c': -1}, N=N, dynamics=dynamics, debug=DEBUG)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))

# Results
# xe__0 0
# ve__0 0
# xl__0 50
# vl__0 0
# p 0.99375
# c 5.625
# b_a 5
# b_g 0.0259274
# The set of optimal parameter values is {'p': 0.99375, 'c': 5.625} with the cost -105.0 and the robustness estimate 0.025927381851941433.
# Time elaspsed for MILP: 433.7607808113098 [seconds].

## ACC Comfort 
# Create acc contract for comfort 
acc_comfort = contract('acc_comfort')
acc_comfort.add_deter_vars(['xe', 've', 'xl', 'vl', 'ae', 'dm', 'vrm', 'vem'],
    bounds=[[0, 50], [0, 30], [0, 50], [0, 30], [-3, 2], [-100, 100], [-30, 30], [0, 30]])

acc_comfort.add_param_vars(['p', 'c'], bounds=[[0.9, 1], [-30, 0]]) # Design space for safety specification

acc_comfort.set_assume(
    '(xl - xe >= {} + {}*ve) & (ve - 5 <= vl) & (vl <= ve + 5)'.format(d_safe, tau) # Safety assumptions
)

acc_comfort.set_guaran(
    'G[0,20] (P[p] (c <= ae))' # Comfort guarantee
)

acc_comfort.checkSat()
acc_comfort.printInfo()
# input()

# Find parameters for acc comfort contract: 
start = time.time()
if INIT:
    opt_dict = acc_comfort.find_opt_param({'p': -100, 'c': -1}, N=N, dynamics=dynamics, init_conditions=init_conditions, debug=DEBUG)
else:
    opt_dict = acc_comfort.find_opt_param({'p': -100, 'c': -1}, N=N, dynamics=dynamics, debug=DEBUG)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))

# Results
# xe__0 0
# ve__0 0
# xl__0 50
# vl__0 0
# p 0.99375
# c -7.5
# b_a 5
# b_g 0.348809
# The set of optimal parameter values is {'p': 0.99375, 'c': -7.5} with the cost -91.875 and the robustness estimate 0.3488093706746431.
# Time elaspsed for MILP: 444.9479172229767 [seconds].