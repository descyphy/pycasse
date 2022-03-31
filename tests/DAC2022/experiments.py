from pystl import *
import time

# Num of maximum deterministic contracts and parameterized contracts
N = 10
M = 3

# Initialize a contract list
contract_list = []

# Create deterministic contracts
for i in range(1, N+1):
    tmp_contract = contract('c{}'.format(i))
    tmp_contract.add_deter_vars(['x'], bounds=[[1,500]])
    tmp_contract.add_nondeter_vars(['w{}'.format(i)],  
        mean = [0],
        cov = [[0.02**2]])
    tmp_contract.set_assume('x <= 300')
    tmp_contract.set_guaran('P[0.95] (w{} <= 0.2)'.format(i)) 
    tmp_contract.checkSat()
    contract_list.append(tmp_contract)

# Create parameterized contracts
for i in range(1, M+1):
    tmp_contract = contract('param_c{}'.format(i))
    tmp_contract.add_deter_vars(['x'], bounds=[[1,500]])
    # tmp_contract.add_param_vars(['p{}'.format(i), 'sigma{}'.format(i), 'x_max{}'.format(i)], bounds = [[0.8, 1], [0.01, 0.3], [100, 500]])
    tmp_contract.add_param_vars(['p{}'.format(i), 'sigma{}'.format(i)], bounds = [[0.8, 1], [0.01, 0.3]])
    tmp_contract.add_nondeter_vars(['param_w{}'.format(i)],  
        mean = [0],
        cov = [['sigma{}^2'.format(i)]])
    tmp_contract.set_assume('x <= 250')
    # tmp_contract.set_assume('x <= x_max{}'.format(i))
    tmp_contract.set_guaran('P[p{}] (param_w{} <= 0.1)'.format(i, i)) 
    tmp_contract.checkSat()
    contract_list.append(tmp_contract)

# Build a contract c
c = contract('c') # Create a contract c
c.add_deter_vars(['x'], bounds=[[1,500]])
c.add_param_vars(['p{}'.format(i) for i in range(1, M+1)] + ['sigma{}'.format(i) for i in range(1, M+1)], bounds = [[0.8, 1]]*M + [[0.01, 0.3]]*M)
tmp_cov = [] # Create a cov matrix for the contract c
for i in range(1, N+1):
    tmp_list = [0]*(N+M)
    tmp_list[i-1] = 0.02**2
    tmp_cov.append(tmp_list)
for j in range(1, M+1):
    tmp_list = [0]*(N+M)
    tmp_list[i+j-1] = 'sigma{}^2'.format(j)
    tmp_cov.append(tmp_list)
c.add_nondeter_vars(['w{}'.format(i) for i in range(1, N+1)] + ['param_w{}'.format(i) for i in range(1, M+1)],  
    mean = [0]*(N+M),
    cov = tmp_cov) # Set nondeterministic uncontrolled variables

# Set assumptions and guarantees for the contract c
c.set_assume('x <= 250') # Set/define the assumptions
expression = ''
for i in range(1, N+1):
    expression += ' + w{}'.format(i)
for j in range(1, M+1):
    expression += ' + param_w{}'.format(j)
c.set_guaran('P[0.99] ({} <= {})'.format(expression[3:len(expression)], 0.05*(N+M))) # Set/define the guarantees
c.checkSat()  # Saturate c
# c.printInfo()

# Compose the contracts
composed = composition(contract_list, mode = 'simple')
# composed.printInfo()

# Find optimal parameters
start = time.time()
weights = {}
for i in range(1, M+1):
    weights['p{}'.format(i)] = -1
    weights['sigma{}'.format(i)] = -10
    weights['x_max{}'.format(i)] = 1
print(weights)
composed.find_opt_refine_param(c, weights, N=200)
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))