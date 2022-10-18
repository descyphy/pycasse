from pycasse import *
import time

DEBUG = False

# Num of partitions, maximum deterministic contracts, and parameterized contracts
# num_partition = 200
# N = 1 # With 1 unknown and num_partition = 10, 100, 200, 500: 0.7451s, 5.6106s, 7.1187s, 16.1808s, 34.0112s
# # N = 10 # With 1 unknown and num_partition = 10, 100, 200, 500: 0.8985s, 6.4869s, 14.1148s, 42.3966s, 75.7964s
# # N = 50 # With 1 unknown and num_partition = 10, 100, 200, 500: 2.6917s, 20.0897s, 37.4683s, 105.0272s, 246.9338s
# # N = 100 # With 1 unknown and num_partition = 10, 100, 200, 500: 4.7572s, 44.6616s, 78.8073s, 208.2345s, 402.8248s
# # N = 200 # With 1 unknown and num_partition = 10, 100, 200, 500: 14.9383s, 112.6803s, 271.5419s, 506.1686s, 1024.7433s
# M = 1
# # M = 2 # With 100 known and num_partition = 10, 100, 200, 500: *1.7474s, 33.5958s, 72.3962s, 183.9817s, 365.8101s
# # M = 3 # With 100 known and num_partition = 10, 100, 200, 500: *1.5163s, *36.8144s, 83.3322s, 191.7921s, 406.4198s
# # M = 4 # With 100 known and num_partition = 10, 100, 200, 500: *0.6852s, *0.7487s, *0.6848s, *144.9243s, 384.5689s
nums_partition = [10, 100, 200, 500, 1000]
Ns = [1, 10, 50, 100, 200]
Ls = [1, 2, 3, 4]

for L in Ls:
    for N in Ns:
        for num_partition in nums_partition:
            # Init printouts
            print("===========================================")
            print("Initiating parameter synthesis with {} specified sensors, {} unspecified sensors, and {} partitions.".format(N, L, num_partition))

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

            # Create parameterized contracts
            for i in range(1, L+1):
                tmp_contract = contract('param_c{}'.format(i))
                tmp_contract.add_deter_vars(['d'], bounds=[[1,500]])
                tmp_contract.add_param_vars(['p{}'.format(i), 'sigma{}'.format(i)], bounds = [[0.8, 1], [0.2, 2]])
                tmp_contract.add_nondeter_vars(['param_n{}'.format(i)],  
                    mean = [0],
                    cov = [['sigma{}^2'.format(i)]])
                tmp_contract.set_assume('d <= 250')
                tmp_contract.set_guaran('P[p{}] (param_n{} <= 1)'.format(i, i)) 
                tmp_contract.checkSat()
                contract_list.append(tmp_contract)

            # Build a contract c
            c = contract('c') # Create a contract c
            c.add_deter_vars(['d'], bounds=[[1,500]])
            c.add_param_vars(['p{}'.format(i) for i in range(1, L+1)] + ['sigma{}'.format(i) for i in range(1, L+1)], bounds = [[0.8, 1]]*L + [[0.1, 2]]*L)
            tmp_cov = [] # Create a cov matrix for the contract c
            for i in range(1, N+1):
                tmp_list = [0]*(N+L)
                tmp_list[i-1] = 0.5**2
                tmp_cov.append(tmp_list)
            for j in range(1, L+1):
                tmp_list = [0]*(N+L)
                tmp_list[i+j-1] = 'sigma{}^2'.format(j)
                tmp_cov.append(tmp_list)
            c.add_nondeter_vars(['n{}'.format(i) for i in range(1, N+1)] + ['param_n{}'.format(i) for i in range(1, L+1)],  
                mean = [0]*(N+L),
                cov = tmp_cov) # Set nondeterministic uncontrolled variables

            # Set assumptions and guarantees for the contract c
            c.set_assume('d <= 250') # Set/define the assumptions
            expression = ''
            for i in range(1, N+1):
                expression += ' + n{}'.format(i)
            for j in range(1, L+1):
                expression += ' + param_n{}'.format(j)
            c.set_guaran('P[0.99] ({} <= {})'.format(expression[3:len(expression)], 1*(N+L))) # Set/define the guarantees
            c.checkSat()  # Saturate c
            if DEBUG:
                c.printInfo()

            # Compose the contracts
            composed = composition(contract_list, mode = 'simple')
            if DEBUG:
                composed.printInfo()

            # Find optimal parameters
            start = time.time()
            weights = {}
            for i in range(1, L+1):
                weights['p{}'.format(i)] = -1
                weights['sigma{}'.format(i)] = -10
            if DEBUG:
                print(weights)
            composed.find_opt_refine_param(c, weights, N=num_partition, debug = DEBUG)
            end = time.time()
            print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
            if DEBUG:
                input()