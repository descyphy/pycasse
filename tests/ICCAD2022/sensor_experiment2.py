from pystl import *
import time

# Num of maximum deterministic contracts and parameterized contracts
# N = 1  # 2.6554s
# N = 10 # 9.1356s
N = 50 # 9.1356s

# Initialize a contract list
contract_list = []

# Create deterministic contracts
for i in range(1, N+1):
    tmp_contract = contract('c{}'.format(i))
    tmp_contract.add_deter_vars(['d'], bounds=[[1,500]])
    tmp_contract.add_nondeter_vars(['n{}'.format(i)],  
        mean = [0],
        cov = [[0.5**2]])
    tmp_contract.set_assume('d <= 300')
    tmp_contract.set_guaran('P[0.95] (n{} <= 2)'.format(i)) 
    tmp_contract.checkSat()
    contract_list.append(tmp_contract)

# Build a contract c
c = contract('c') # Create a contract c
c.add_deter_vars(['d'], bounds=[[1,500]])
c.add_param_vars(['p', 'c'], bounds = [[0.8, 1], [0, 2]])
tmp_cov = [] # Create a cov matrix for the contract c
for i in range(1, N+1):
    tmp_list = [0]*(N)
    tmp_list[i-1] = 0.02**2
    tmp_cov.append(tmp_list)
c.add_nondeter_vars(['n{}'.format(i) for i in range(1, N+1)],  
    mean = [0]*(N),
    cov = tmp_cov) # Set nondeterministic uncontrolled variables

# Set assumptions and guarantees for the contract c
c.set_assume('d <= 250') # Set/define the assumptions
expression = ''
for i in range(1, N+1):
    expression += ' + n{}'.format(i)
c.set_guaran('P[p] ({} <= {}*c)'.format(expression[3:len(expression)], N)) # Set/define the guarantees
c.checkSat()  # Saturate c
c.printInfo()

# Compose the contracts
composed = composition(contract_list, mode = 'simple')
composed.printInfo()
input()

# Find optimal parameters
start = time.time()
composed.find_opt_refine_param(c, {'p': -1, 'c': 1}, N=200)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))