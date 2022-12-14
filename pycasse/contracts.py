import gurobipy as gp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from copy import deepcopy
from collections import deque
from gurobipy import GRB
from pycasse.core import MILPSolver
from pycasse.parser import Parser

M = 10**4
EPS = 10**-4
parser = Parser()

class contract:
    """
    A contract class for defining a contract.
    """
    __slots__ = ('id', 'deter_var_list', 'deter_var_types', 'deter_var_bounds', 'nondeter_var_list', 'nondeter_var_types', 'nondeter_var_mean', 'nondeter_var_cov', 'param_var_list', 'param_var_types', 'param_var_bounds', 'assumption_str', 'assumption', 'guarantee_str', 'guarantee', 'sat_guarantee_str', 'sat_guarantee', 'isSat', 'objectives')

    def __init__(self, id = ''):
        """ Constructor method """
        self.id = id
        self.reset()

    def reset(self):
        """ Resets the contract """
        self.deter_var_list       = []
        self.deter_var_types      = []
        self.deter_var_bounds     = []
        self.param_var_list       = []
        self.param_var_types      = []
        self.param_var_bounds     = []
        self.nondeter_var_list    = []
        self.nondeter_var_types   = []
        self.nondeter_var_mean    = []
        self.nondeter_var_cov     = [[]]
        self.assumption_str       = 'True'
        self.assumption           = parser('True')[0][0]
        self.guarantee_str        = 'False'
        self.guarantee            = parser('False')[0][0]
        self.sat_guarantee_str    = 'False'
        self.sat_guarantee        = parser('False')[0][0]
        self.isSat                = False
        self.objectives           = []

    def add_deter_vars(self, var_names, dtypes = None, bounds = None):
        """
        Adds deterministic variables and their information to the contract. 

        :param var_names: A list of names for controlled variables
        :type var_names: list
        :param dtypes: A list of variable types for controlled variables, each entry can be either `BINARY`, `INTEGER`, or `CONTINUOUS`. If None, defaults to `CONTINUOUS`.
        :type dtypes: list, optional
        :param bounds: A list of lower and upper bounds for controlled variables, defaults to `[-10^4,10^4]` for `CONTINUOUS` and `INTEGER` variables and `[0,1]` for a `BINARY` variable, defaults to `None`.
        :type bounds: list, optional
        """
        # For all variables, construct a variable class
        for i, name in enumerate(var_names):
            self.deter_var_list.append(name)
            self.deter_var_types.append(dtypes[i] if dtypes is not None else 'CONTINUOUS')
            self.deter_var_bounds.append(bounds[i] if bounds is not None else [-M, M])

    def add_nondeter_vars(self, var_names, mean, cov, dtypes = None):
        """
        Adds nondeterministic variables and their information to the contract. 

        :param var_names: A list of names for nondeterministic variables
        :type var_names: list
        :param mean: A mean vector of nondeterministic variables
        :type mean: list of floats
        :param cov: A covariance matrix of nondeterministic variables
        :type cov: list of lists of floats
        :param dtype: Distribution types for nondeterministic variables, can only support `GAUSSIAN`, defaults to `None`.
        :type dtype: list of strs, optional
        """
        # Check the dimensions of mean and covariance matrix
        assert(len(mean) == len(var_names))
        assert(len(cov) == len(var_names))
        assert(len(cov[0]) == len(var_names))

        self.nondeter_var_mean = mean
        self.nondeter_var_cov = cov

        # For all variables, construct a variable class
        for i, name in enumerate(var_names):
            self.nondeter_var_list.append(name)
            self.nondeter_var_types.append(dtypes[i] if dtypes is not None else 'CONTINUOUS')

    def add_param_vars(self, param_names, dtypes = None, bounds = None):
        """
        Adds parameterss and their information to the contract.

        :param param_names  : A list of names for parameters
        :type  param_names  : list
        :param dtypes       : A list of variable types for parameter variables, each entry can be either "BINARY", "INTEGER", or "CONTINUOUS", defaults to "CONTINUOUS"
        :type  dtypes       : list, optional
        :param bounds       : A list of lower and upper bounds for controlled variables, defaults to `[-10^4,10^4]` for "CONTINUOUS" and "INTEGER" variable and `[0,1]` for "BINARY"
        :type  bounds       : list, optional
        """
        # For all variables, construct a variable class
        for i, name in enumerate(param_names):
            self.param_var_list.append(name)
            self.param_var_types.append(dtypes[i] if dtypes is not None else 'CONTINUOUS')
            self.param_var_bounds.append(bounds[i] if bounds is not None else [-M, M])

    def set_assume(self, assumption):
        """
        Sets the assumption of the contract.

        :param assumption: An STL or StSTL formula which characterizes the assumption set of the contract
        :type assumption: str
        """
        self.assumption_str = ' '.join(assumption.split())
        self.assumption = parser(self.assumption_str)[0][0]
        for variable in self.assumption.variables:
            if variable != 1 and not (variable in self.deter_var_list or variable in self.nondeter_var_list or variable in self.param_var_list):
                raise ValueError("Variable {} not in the contract variables.".format(variable))

    def set_guaran(self, guarantee):
        """
        Sets the guarantee of the contract.

        :param guarantee: An STL or StSTL formula which characterizes the guarantee set of the contract
        :type guarantee: str
        """
        self.guarantee_str = ' '.join(guarantee.split())
        self.guarantee = parser(self.guarantee_str)[0][0]
        for variable in self.guarantee.variables:
            if variable != 1 and not (variable in self.deter_var_list or variable in self.nondeter_var_list or variable in self.param_var_list):
                raise ValueError("Variable {} not in the contract variables.".format(variable))
    
    def set_sat_guaran(self, sat_guarantee):
        """
        Sets the guarantee of the contract.

        :param guarantee: An STL or StSTL formula which characterizes the guarantee set of the contract
        :type guarantee: str
        """
        self.sat_guarantee_str = ' '.join(sat_guarantee.split())
        self.sat_guarantee = parser(self.sat_guarantee_str)[0][0]
        for variable in self.guarantee.variables:
            if variable != 1 and not (variable in self.deter_var_list or variable in self.nondeter_var_list or variable in self.param_var_list):
                raise ValueError("Variable {} not in the contract variables.".format(variable))

    def saturate(self):
        """
        Saturates the contract. 
        """
        if not self.isSat:
            self.checkSat()

    def checkSat(self):
        """ Checks whether the contract is saturated or not. If not saturated, saturate the contract. """
        if not self.isSat:
            if self.assumption_str == 'True':
                self.sat_guarantee_str = self.guarantee_str
            elif self.assumption_str == 'False':
                self.sat_guarantee_str = 'True'
            else:
                self.sat_guarantee_str = '({}) -> ({})'.format(self.assumption_str, self.guarantee_str)
            self.sat_guarantee = parser(self.sat_guarantee_str)[0][0]
            self.isSat = True
    
    def checkCompat(self, dynamics = None, init_conditions = [], print_sol=False):
        """ Checks compatibility of the contract.

        :param dynamics: A dictionary describing the dynamics of the component/system, defaults to `None`
        :type dynamics: dict, optional
        :param init_conditions: A list of str describing the initial conditions of the system or the component, defaults to '[]'
        :type init_conditions: list of str, optional
        :param print_sol: If `True`, prints the behavior which shows the compatibility of the component/system, defaults to `False`
        :type print_sol: bool, optional
        :return: `True` if compatible, `False` otherwise.
        :rtype: bool
        """
        # Build a MILP Solver
        print("====================================================================================")
        print("Checking compatibility of the contract {}...".format(self.id))
        solver = MILPSolver()

        # Add the contract and assumption constraints to the solver
        self.checkSat()
        solver.add_contract(self)
        if dynamics is not None:
            solver.add_dynamics(x=dynamics['x'], u=dynamics['u'], w=dynamics['w'], A=dynamics['A'], B=dynamics['B'], C=dynamics['C'])
        for init_condition in init_conditions:
            solver.add_init_condition(init_condition)
        solver.add_constraint(self.assumption)

        # Solve the problem
        solved = solver.solve()

        # Print the solution
        if solved:
            print("Contract {} is compatible.\n".format(self.id))
            if print_sol:
                print("Printing a behavior that satisfies the assumptions of the contract {}...".format(self.id))
                solver.print_solution()
        elif not solved:
            print("Contract {} is not compatible.\n".format(self.id))

        return solved
    
    def checkConsis(self, dynamics = None, init_conditions = [], print_sol=False):
        """ Checks consistency of the contract 

        :param dynamics: A dictionary describing the dynamics of the component/system, defaults to `None`
        :type dynamics: dict, optional
        :param init_conditions: A list of str describing the initial conditions of the system or the component, defaults to '[]'
        :type init_conditions: list of str, optional
        :param print_sol: If `True`, prints the behavior which shows the consistency of the component/system, defaults to `False`
        :type print_sol: bool, optional
        :return: `True` if compatible, `False` otherwise.
        :rtype: bool
        """
        # Build a MILP Solver
        print("====================================================================================")
        print("Checking consistency of the contract {}...".format(self.id))
        solver = MILPSolver()

        # Add the contract and guarantee constraints to the solver
        self.checkSat()
        solver.add_contract(self)
        if dynamics is not None:
            solver.add_dynamics(x=dynamics['x'], u=dynamics['u'], w=dynamics['w'], A=dynamics['A'], B=dynamics['B'], C=dynamics['C'])
        for init_condition in init_conditions:
            solver.add_init_condition(init_condition)
        solver.add_constraint(self.sat_guarantee)

        # Solve the problem
        solved = solver.solve()

        # Print the solution
        if solved:
            print("Contract {} is consistent.\n".format(self.id))
            if print_sol:
                print("Printing a behavior that satisfies the saturated guarantees of the contract {}...".format(self.id))
                solver.print_solution()
        elif not solved:
            print("Contract {} is not consistent.\n".format(self.id))

        return solved
    
    def checkFeas(self, dynamics = None, init_conditions = [], print_sol=False):
        """ Checks feasibility of the contract.
        
        :param dynamics: A dictionary describing the dynamics of the component/system, defaults to `None`
        :type dynamics: dict, optional
        :param init_conditions: A list of str describing the initial conditions of the system or the component, defaults to '[]'
        :type init_conditions: list of str, optional
        :param print_sol: If `True`, prints the behavior which shows the feasibility of the component/system, defaults to `False`
        :type print_sol: bool, optional
        :return: `True` if compatible, `False` otherwise.
        :rtype: bool
        """
        # Build a MILP Solver
        print("====================================================================================")
        print("Checking feasibility of the contract {}...".format(self.id))
        solver = MILPSolver()

        # Add the contract and assumption constraints to the solver
        self.checkSat()
        solver.add_contract(self)
        if dynamics is not None:
            solver.add_dynamics(x=dynamics['x'], u=dynamics['u'], w=dynamics['w'], A=dynamics['A'], B=dynamics['B'], C=dynamics['C'])
        for init_condition in init_conditions:
            solver.add_init_condition(init_condition)
        solver.add_constraint(parser("({}) & ({})".format(self.assumption_str, self.guarantee_str))[0][0])

        # Solve the problem
        solved = solver.solve()

        # Print the solution
        if solved:
            print("Contract {} is feasible.\n".format(self.id))
            if print_sol:
                print("Printing a behavior that satisfies both the assumptions and guarantees of the contract {}...".format(self.id))
                solver.print_solution()
        elif not solved:
            print("Contract {} is not feasible.\n".format(self.id))

        return solved
    
    def checkRefine(self, contract2refine, dynamics = None, init_conditions = [], print_sol=False):
        """
        Checks whether the self contract refines another contract.

        :param contract2refine: A contract to refine.
        :type contract2refine: :class:`pycasse.contracts.contract`
        :param dynamics: A dictionary describing the dynamics of the component/system, defaults to `None`
        :type dynamics: dict, optional
        :param init_conditions: A list of str describing the initial conditions of the system or the component, defaults to '[]'
        :type init_conditions: list of str, optional
        :param print_sol: If `True`, prints the behavior which violates the refinement relationship, defaults to `False`
        :type print_sol: bool, optional
        :return: `True` if the refinement relationship holds, `False` otherwise.
        :rtype: bool
        """
        print("====================================================================================")
        print("Checking whether contract {} refines contract {}...".format(self.id, contract2refine.id))
        
        # Check saturation of the contracts
        self.checkSat()
        contract2refine.checkSat()

        # Build a contract for checking refinement
        refinement_contract = deepcopy(self)
        refinement_contract.id = 'refinement_check'
        # refinement_contract.printInfo()
        # contract2refine.printInfo()

        # Merge variables
        merge_variables(refinement_contract, contract2refine)
        
        # Find assumption and guarantee
        refinement_contract.set_assume("(!({})) & ({})".format(self.assumption_str, contract2refine.assumption_str))
        refinement_contract.set_guaran("(!({})) & ({})".format(contract2refine.sat_guarantee_str, self.sat_guarantee_str))
        # refinement_contract.set_guaran("(!({})) & ({})".format(contract2refine.guarantee_str, self.guarantee_str))
        # refinement_contract.printInfo()

        # Build a MILP Solver
        solver = MILPSolver()

        # Add constraints for refinement condition for assumptions
        print("Checking assumptions condition for refinement...")
        self.checkSat()
        solver.add_contract(self)
        if dynamics is not None:
            solver.add_dynamics(x=dynamics['x'], u=dynamics['u'], w=dynamics['w'], A=dynamics['A'], B=dynamics['B'], C=dynamics['C'])
        for init_condition in init_conditions:
            solver.add_init_condition(init_condition)
        solver.add_constraint(refinement_contract.assumption)
    
        # Check refinement condition for assumptions
        solved = solver.solve()
    
        # Print the counterexample
        if solved:
            print("Assumptions condition for refinement violated. Contract {} does not refine contract {}.\n".format(self.id, contract2refine.id))
            if print_sol:
                print("Printing a counterexample which violates assumptions condition for refinement...")
                solver.print_solution()
            return False

        # Resets a MILP Solver
        solver.reset()

        # Add constraints for refinement condition for guarantees
        print("Checking guarantees condition for refinement...")
        self.checkSat()
        solver.add_contract(contract2refine)
        if dynamics is not None:
            solver.add_dynamics(x=dynamics['x'], u=dynamics['u'], w=dynamics['w'], A=dynamics['A'], B=dynamics['B'], C=dynamics['C'])
        for init_condition in init_conditions:
            solver.add_init_condition(init_condition)
        solver.add_constraint(refinement_contract.guarantee)

        # Check refinement condition for guarantees
        solved = solver.solve()

        # Print the counterexample
        if solved:
            print("Guarantees condition for refinement violated. Contract {} does not refine contract {}.\n".format(self.id, contract2refine.id))
            if print_sol:
                print("Printing a counterexample which violates guarantees condition for refinement...")
                solver.print_solution()
            return False

        print("Contract {} refines {}.\n".format(self.id, contract2refine.id))
        return True
    
    def find_opt_refine_param(self, contract2refine, weights, N=100, dynamics = None, init_conditions = [], debug=False):
        """
        Find an optimal set of parameters given an objective function such that the self contract refine the contract2refine.

        :param contract2refine: A contract to refine.
        :type contract2refine: :class:`pycasse.contracts.contract`
        :param weights: A dictionary containing information on the weight of each parameter
        :type weights: dict
        :param N: The maximum number of partitions, defaults to `100`
        :type N: int, optional
        :param dynamics: A dictionary describing the dynamics of the component/system, defaults to `None`
        :type dynamics: dict, optional
        :param init_conditions: A list of str describing the initial conditions of the system or the component, defaults to '[]'
        :type init_conditions: list of str, optional
        :param debug: If `True`, prints useful information for debugging the function, defaults to `False`
        :type debug: bool, optional
        """
        # Check saturation of the contracts
        self.checkSat()
        if debug:
            self.printInfo()
        contract2refine.checkSat()

        # Build a contract for checking refinement
        refinement_contract = deepcopy(self)
        refinement_contract.id = 'refinement_check'
        if debug:
            contract2refine.printInfo()

        # Merge variables
        merge_variables(refinement_contract, contract2refine)
        
        # Find assumption and guarantee
        # refinement_contract.set_assume(contract2refine.assumption_str)
        refinement_contract.set_assume("({}) & ({})".format(contract2refine.assumption_str, self.assumption_str))
        # refinement_contract.set_assume("({}) -> ({})".format(contract2refine.assumption_str, self.assumption_str))
        # refinement_contract.set_assume("(({}) -> ({})) & ({})".format(contract2refine.assumption_str, self.assumption_str, self.guarantee_str))
        # refinement_contract.set_assume("(({}) -> ({})) & ({})".format(contract2refine.assumption_str, self.assumption_str, self.sat_guarantee_str))

        refinement_contract.set_guaran("({}) & ({})".format(self.guarantee_str, contract2refine.guarantee_str))
        # refinement_contract.set_guaran("({}) & ({}) & ({})".format(self.assumption_str, self.guarantee_str, contract2refine.guarantee_str))
        # refinement_contract.set_guaran("({}) & ({}) & ({}) & ({})".format(contract2refine.assumption_str, self.assumption_str, self.guarantee_str, contract2refine.guarantee_str))
        # refinement_contract.set_guaran("({}) -> ({})".format(self.sat_guarantee_str, contract2refine.sat_guarantee_str))
        
        refinement_contract.set_sat_guaran("({}) & ({})".format(self.guarantee_str, contract2refine.guarantee_str))
        # refinement_contract.set_sat_guaran("({}) & ({}) & ({})".format(self.assumption_str, self.guarantee_str, contract2refine.guarantee_str))
        # refinement_contract.set_sat_guaran("({}) & ({}) & ({}) & ({})".format(contract2refine.assumption_str, self.assumption_str, self.guarantee_str, contract2refine.guarantee_str))
        # refinement_contract.set_sat_guaran("({}) -> ({})".format(self.sat_guarantee_str, contract2refine.sat_guarantee_str))
        refinement_contract.isSat = True
        if debug:
            refinement_contract.printInfo()

        # Find an optimal set of parameters
        refinement_contract.find_opt_param(weights, N=N, dynamics=dynamics, init_conditions=init_conditions, debug=debug)
        
    def find_opt_param(self, weights, N=100, dynamics = None, init_conditions = [], debug=False):
        """
        Find an optimal set of parameters for a contract given an objective function.

        :param weights: A dictionary containing information on the weight of each parameter
        :type weights: dict
        :param N: The maximum number of partitions, defaults to `100`
        :type N: int, optional
        :param dynamics: A dictionary describing the dynamics of the component/system, defaults to `None`
        :type dynamics: dict, optional
        :param init_conditions: A list of str describing the initial conditions of the system or the component, defaults to '[]'
        :type init_conditions: list of str, optional
        :param debug: If `True`, prints useful information for debugging the function, defaults to `False`
        :type debug: bool, optional
        :return: A dictionary of the optimal parameter values
        :rtype: dict
        """
        print("Finding an optimal set of parameters for contract {}...".format(self.id))

        # Initialize the initial bounds, SAT, UNSAT, and UNDETERMINED sets
        bounds = deepcopy(self.param_var_bounds)
        param_SAT = []
        param_UNSAT = []
        param_UNDET = [deepcopy(self.param_var_bounds)]
        param_num = len(bounds)

        # Pre-partition the parameter space for probability thresholds
        count = 0
        for var_name in self.param_var_list:
            if 'p' in var_name:
                for tmp_partition in param_UNDET:
                    if tmp_partition[count][0] < 0.5:
                        # Add another partition
                        additional_partition = deepcopy(tmp_partition)
                        additional_partition[count][0] = 0.5
                        param_UNDET.append(additional_partition)

                        # Modify the current partition
                        tmp_partition[count][1] = 0.5
            count += 1
            
        def findPartitionType(partition, dyn, inits, debug=False):
            # Print the current partition
            if debug:
                print("============================================================")
                print(partition)

            # Update the bounds 
            self.param_var_bounds = partition
            # self.printInfo()

            # Build a Solver
            realMILPsolver = MILPSolver(mode="Quantitative")
            # MILPsolver = MILPSolver()
            realMILPsolver.add_contract(self)
            if dyn is not None:
                realMILPsolver.add_dynamics(x=dyn['x'] if 'x' in dyn.keys() else [], 
                u=dyn['u'] if 'u' in dyn.keys() else [], 
                z=dyn['z'] if 'z' in dyn.keys() else [], 
                A=dyn['A'] if 'A' in dyn.keys() else None, 
                B=dyn['B'] if 'B' in dyn.keys() else None, 
                C=dyn['C'] if 'C' in dyn.keys() else None, 
                D=dyn['D'] if 'D' in dyn.keys() else None, 
                E=dyn['E'] if 'E' in dyn.keys() else None, 
                Q=dyn['Q'] if 'Q' in dyn.keys() else None, 
                R=dyn['R'] if 'R' in dyn.keys() else None)

            for init_condition in inits:
                realMILPsolver.add_init_condition(init_condition)
            realMILPsolver.add_constraint(self.assumption, name='b_a')
            realMILPsolver.add_constraint(self.guarantee, hard=False, name='b_g')
            # realMILPsolver.add_constraint(self.sat_guarantee, hard=False, name='b_g')
            # tmp_parsetree = '({}) & ({})'.format(self.assumption_str, self.sat_guarantee_str)
            # tmp_parsetree = parser(tmp_parsetree)[0][0]
            # realMILPsolver.add_constraint(tmp_parsetree, hard=False, name='b_g')

            # Solve the problem
            realMILPsolver.set_objective(sense='minimize')
            if not realMILPsolver.solve():
                if debug:
                    print("UNSAT partition!")
                # input()
                return 0, 1
            else:
                # for v in realMILPsolver.model.getVars():
                #     # print('%s %g' % (v.varName, v.x))
                #     if 'b' not in v.varName:
                #         print('%s %g' % (v.varName, v.x))
                #     elif v.varName in ('b_a', 'b_g'):
                #         print('%s %g' % (v.varName, v.x))
                objective_val1 = realMILPsolver.soft_constraint_vars[0].x
                if debug:
                    print(objective_val1)
                if objective_val1 >= 0:
                    if debug:
                        print("SAT partition!")
                    return objective_val1, 0
                else:
                    realMILPsolver.set_objective(sense='maximize')
                    realMILPsolver.solve()
                    # for v in realMILPsolver.model.getVars():
                    #     # print('%s %g' % (v.varName, v.x))
                    #     if 'b' not in v.varName:
                    #         print('%s %g' % (v.varName, v.x))
                    #     elif v.varName in ('b_a', 'b_g'):
                    #         print('%s %g' % (v.varName, v.x))
                    objective_val2 = realMILPsolver.soft_constraint_vars[0].x
                    if debug:
                        print(objective_val2)
                    if objective_val2 <= -EPS:
                        if debug:
                            print("UNSAT partition!")
                        return objective_val2, 1
                    else: 
                        if debug:
                            print("UNDET partition!")
                        # print(objective_val2-objective_val1)
                        return objective_val2-objective_val1, 2

        def paramSpacePartition(partition):
            """
            Partitions the given partition.

            :param partition: [description]
            :type partition: [type]
            """
            # Initialize the partition list
            prev_partition_list = [partition]

            # TODO: Add more partition methods
            # Binary partition
            for i in range(len(partition)):
                curr_partition_list = []
                for tmp_partition in prev_partition_list:
                    # Add another partition
                    additional_partition = deepcopy(tmp_partition)
                    binary_partition_num = (additional_partition[i][0]+additional_partition[i][1])/2
                    additional_partition[i][1] = binary_partition_num
                    curr_partition_list.append(additional_partition)

                    # Modify the current partition
                    tmp_partition[i][0] = binary_partition_num
                    curr_partition_list.append(tmp_partition)
                
                prev_partition_list = curr_partition_list
            
            return prev_partition_list

        # Find SAT and UNSAT partitions
        while len(param_UNDET) != 0 and len(param_SAT)+len(param_UNSAT)+len(param_UNDET)-1+2**param_num <= N:
            tmp_partition = param_UNDET.pop(0)
            obj_val, partition_type = findPartitionType(tmp_partition, dynamics, init_conditions, debug=debug)
            if partition_type == 0: # SAT partition
                param_SAT.append(tmp_partition)
            elif partition_type == 1: # UNSAT partition
                param_UNSAT.append(tmp_partition)
            elif partition_type == 2: # UNDET partition
                for partitioned_partition in paramSpacePartition(tmp_partition):
                    param_UNDET.append(partitioned_partition)

        # Double-check UNDET partitions
        tmp_param_UNDET = deepcopy(param_UNDET)
        # print(tmp_param_UNDET)
        # input()
        param_UNDET = []
        for tmp_partition in tmp_param_UNDET:
            # print(param_UNDET)
            # print(len(param_SAT)+len(param_UNSAT)+len(param_UNDET))
            obj_val, partition_type = findPartitionType(tmp_partition, dynamics, init_conditions, debug=debug)
            if partition_type == 0: # SAT partition
                param_SAT.append(tmp_partition)
            elif partition_type == 1: # UNSAT partition
                param_UNSAT.append(tmp_partition)
            elif partition_type == 2: # UNDET partition
                param_UNDET.append(tmp_partition)

        # print(param_SAT)
        # print(param_UNSAT)
        # print(param_UNDET)
        # print(sorted_param_UNDET)

        def findMinCost(weights, partition):
            # Initialize Gurobi model
            model = gp.Model()
            model.setParam("OutputFlag", 0)
            
            # Add variables
            variables = []
            for i, var_name in enumerate(self.param_var_list):
                variables.append(model.addVar(lb=partition[i][0], ub=partition[i][1], vtype=GRB.CONTINUOUS, name=var_name))
            model.update()

            # Add objective
            obj = 0
            for variable in variables:
                obj += weights[variable.varName]*variable
            model.setObjective(obj, GRB.MINIMIZE)

            # Solve the optimization problem
            model.optimize()

            # Fetch cost and optimal set of parameters
            obj = model.getObjective()
            optimal_params = {}
            for var_name in self.param_var_list:
                optimal_params[var_name] = model.getVarByName(var_name).x

            # print(obj.getValue())
            # print(optimal_params)
            
            return obj.getValue(), optimal_params

        # Find the optimal set of parameters
        minCost = 10**4
        optimal_params = {}
        optimal_partition = None
        for partition in param_SAT:
            tmp_minCost, tmp_optimal_params = findMinCost(weights, partition)
            if tmp_minCost < minCost:
                minCost = tmp_minCost
                optimal_params = tmp_optimal_params
                optimal_partition = deepcopy(partition)
        
        if not not optimal_partition:
            # Find the robustness estimate for the optimal parameters
            # Build a Solver
            realMILPsolver2 = MILPSolver(mode="Quantitative")
            self.param_var_bounds = optimal_partition
            realMILPsolver2.add_contract(self)
            if dynamics is not None:
                realMILPsolver2.add_dynamics(x=dynamics['x'] if 'x' in dynamics.keys() else [], 
                u=dynamics['u'] if 'u' in dynamics.keys() else [], 
                z=dynamics['z'] if 'z' in dynamics.keys() else [], 
                A=dynamics['A'] if 'A' in dynamics.keys() else None, 
                B=dynamics['B'] if 'B' in dynamics.keys() else None, 
                C=dynamics['C'] if 'C' in dynamics.keys() else None, 
                D=dynamics['D'] if 'D' in dynamics.keys() else None, 
                E=dynamics['E'] if 'E' in dynamics.keys() else None, 
                Q=dynamics['Q'] if 'Q' in dynamics.keys() else None, 
                R=dynamics['R'] if 'R' in dynamics.keys() else None)

            for init_condition in init_conditions:
                realMILPsolver2.add_init_condition(init_condition)
            realMILPsolver2.add_constraint(self.assumption, name='b_a')
            realMILPsolver2.add_constraint(self.guarantee, hard=False, name='b_g')
            for k, v in optimal_params.items():
                realMILPsolver2.add_constraint(parser('{} == {}'.format(k,v))[0][0], name='b_{}'.format(k))

            # Solve the problem
            realMILPsolver2.set_objective(sense='minimize')
            realMILPsolver2.solve()

            # Print results
            for v in realMILPsolver2.model.getVars():
                # print('%s %g' % (v.varName, v.x))
                if 'b' not in v.varName:
                    print('%s %g' % (v.varName, v.x))
                elif v.varName in ('b_a', 'b_g'):
                    print('%s %g' % (v.varName, v.x))
            objective_val = realMILPsolver2.soft_constraint_vars[0].x
            print("The set of optimal parameter values is {} with the cost {} and the robustness estimate {}.".format(optimal_params, minCost, objective_val))
        else:
            print("Cannot find a set of optimal parameter values.")


        # Plot SAT, UNSAT, and UNDET regions 
        if len(self.param_var_list) == 2: # If the parameter space is 2D
            _, ax = plt.subplots()
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.xlabel(self.param_var_list[0], fontsize='large')
            plt.ylabel(self.param_var_list[1], fontsize='large')
            # plt.xlabel(r'$p$', fontsize='large')
            # plt.ylabel(r'$\sigma$', fontsize='large')
            # plt.xlabel(r'$p_{s}$', fontsize='large')
            # plt.ylabel(r'$c_{s}$', fontsize='large')
            # plt.xlabel(r'$p_{c}$', fontsize='large')
            # plt.ylabel(r'$c_{c}$', fontsize='large')
            plt.xlim(bounds[0][0], bounds[0][1])
            plt.ylim(bounds[1][0], bounds[1][1])

            # Color SAT regions
            for partition in param_SAT:
                ax.add_patch(patches.Rectangle(
                        (partition[0][0], partition[1][0]),
                        partition[0][1]-partition[0][0], partition[1][1]-partition[1][0],
                        edgecolor = 'black', facecolor = 'lime', fill=True))

            # Color UNSAT regions
            for partition in param_UNSAT:
                ax.add_patch(patches.Rectangle(
                        (partition[0][0], partition[1][0]),
                        partition[0][1]-partition[0][0], partition[1][1]-partition[1][0],
                        edgecolor = 'black', facecolor = 'red', fill=True))

            # Color UNDET regions
            for partition in param_UNDET:
                ax.add_patch(patches.Rectangle(
                        (partition[0][0], partition[1][0]),
                        partition[0][1]-partition[0][0], partition[1][1]-partition[1][0],
                        edgecolor = 'black', facecolor = 'grey', fill=True))

            # Save the figure
            plt.savefig('{}_param_opt.pdf'.format(self.id))
        
        # return optimal_params
        return optimal_params

    def printInfo(self):
        print(str(self))

    def __str__(self):
        """ Prints information of the contract """
        res = ""
        res += "Contract ID: {}\n".format(self.id)
        if len(self.deter_var_list):
            res += "  Deterministic Variables: \n"
            for i, v in enumerate(self.deter_var_list):
                res += "    {}, {}, {}\n".format(v, self.deter_var_types[i], self.deter_var_bounds[i])
        if len(self.nondeter_var_list):
            res += "  Non-Deterministic Variables: \n"
            for i, v in enumerate(self.nondeter_var_list):
                res += "    {}, {}\n".format(v, self.nondeter_var_types[i])
            res += "    mean: {}\n".format(self.nondeter_var_mean)
            res += "    cov: {}\n".format(self.nondeter_var_cov)
        if len(self.param_var_list):
            res += "  Parameteric Variables: \n"
            for i, v in enumerate(self.param_var_list):
                res += "    {}, {}, {}\n".format(v, self.param_var_types[i], self.param_var_bounds[i])
        res += "  Assumption: {}\n".format(self.assumption_str)
        res += "  Guarantee: {}\n".format(self.guarantee_str)
        res += "  Saturated Guarantee: {}\n".format(self.sat_guarantee_str)
        res += "  isSat: {}\n".format(self.isSat)
        return res

    def __repr__(self):
        """ Prints information of the contract """
        res = ""
        res += "Contract ID: {}".format(self.id)
        for v in self.deter_var_list:
            res += "\n"
            res += "\n      ".join(repr(v).splitlines())
        for v in self.nondeter_var_list:
            res += "\n"
            res += "\n      ".join(repr(v).splitlines())
        res += "\n"
        res += "    Assumption: {}\n".format(self.assumption)
        res += "    Guarantee: {}\n".format(self.guarantee)
        res += "    Saturated Guarantee: {}\n".format(self.sat_guarantee)
        res += "    isSat: {}\n".format(self.isSat)
        return res

def conjunction(c_list):
    """ Returns the conjunction of two contracts

    :param c1: A contract c1
    :type c1: :class:`pycasse.contracts.contract`
    :param c2: A contract c2
    :type c2: :class:`pycasse.contracts.contract`
    :return: A conjoined contract
    :rtype: :class:`pycasse.contracts.contract`
    """
    # Initialize variables
    first_contract = True
    assumption_str = ''
    guarantee_str = ''

    if len(c_list) == 1: # If there is only one contract in the list, return that contract
        return c_list[0]
    
    else: # If there are more than two contracts in the list, compose the contracts and return the composed contract
        for c in c_list:
            # Check saturation of c, saturate it if not saturated
            c.checkSat()

            if first_contract:
                # Initialize a conposed contract object
                conjoined = deepcopy(c)
                conjoined.id = '{}'.format(c.id)
                assumption_str += "({})".format(c.assumption_str)
                guarantee_str += "({})".format(c.sat_guarantee_str)

                first_contract = False
            else:
                # Update the conjoined name
                conjoined.id += '^{}'.format(c.id)

                # Merge variables
                merge_variables(conjoined, c)

                # Find assumptions and guarantees
                assumption_str += " | ({})".format(c.assumption_str)
                guarantee_str += " & ({})".format(c.sat_guarantee_str)

        conjoined.set_assume(assumption_str)   
        conjoined.set_guaran(guarantee_str)
        conjoined.set_sat_guaran(guarantee_str)
        conjoined.isSat = True

        return conjoined

def merge(c_list):
    """ Returns the merge of two contracts

    :param c1: A contract c1
    :type c1: :class:`pycasse.contracts.contract.contract`
    :param c2: A contract c2
    :type c2: :class:`pycasse.contracts.contract.contract`
    :return: A merged contract
    :rtype: :class:`pycasse.contracts.contract.contract`
    """
    # Initialize variables
    first_contract = True
    assumption_str = ''
    guarantee_str = ''

    if len(c_list) == 1: # If there is only one contract in the list, return that contract
        return c_list[0]
    
    else: # If there are more than two contracts in the list, compose the contracts and return the composed contract
        for c in c_list:
            # Check saturation of c, saturate it if not saturated
            c.checkSat()

            if first_contract:
                # Initialize a conposed contract object
                merged = deepcopy(c)
                merged.id = '{}'.format(c.id)
                assumption_str += "({})".format(c.assumption_str)
                guarantee_str += "({})".format(c.sat_guarantee_str)

                first_contract = False
            else:
                # Update the conjoined name
                merged.id += '+{}'.format(c.id)

                # Merge variables
                merge_variables(merged, c)

                # Find assumptions and guarantees
                assumption_str += " & ({})".format(c.assumption_str)
                guarantee_str += " & ({})".format(c.sat_guarantee_str)

        merged.set_assume(assumption_str)   
        merged.set_guaran(guarantee_str)
        merged.set_sat_guaran(guarantee_str)
        merged.isSat = True

        return merged

def composition(c_list, mode = 'default'):
    """Returns the composition of the contracts in a list.

    :param c_list: A list of contracts
    :type c_list: list of :class:`pycasse.contracts.contract`
    :return: A composed contract
    :rtype: :class:`pycasse.contracts.contract`
    """
    # Initialize variables
    first_contract = True
    assumption_str1 = ''
    assumption_str2 = ''
    guarantee_str = ''
    sat_guarantee_str = ''

    if len(c_list) == 1: # If there is only one contract in the list, return that contract
        return c_list[0]
    
    else: # If there are more than two contracts in the list, compose the contracts and return the composed contract
        for c in c_list:
            # Check saturation of c, saturate it if not saturated
            c.checkSat()

            if first_contract:
                # Initialize a conposed contract object
                composed = deepcopy(c)
                composed.id = '{}'.format(c.id)
                assumption_str1 += "({})".format(c.assumption_str)
                assumption_str2 += "(!({}))".format(c.sat_guarantee_str)
                guarantee_str += "({})".format(c.guarantee_str)
                sat_guarantee_str += "({})".format(c.sat_guarantee_str)

                first_contract = False
            else:
                # Update the conjoined name
                composed.id += '*{}'.format(c.id)

                # Merge variables
                merge_variables(composed, c)

                # Find assumptions
                assumption_str1 += " & ({})".format(c.assumption_str)
                assumption_str2 += " | (!({}))".format(c.sat_guarantee_str)
                guarantee_str += " & ({})".format(c.guarantee_str)
                sat_guarantee_str += " & ({})".format(c.sat_guarantee_str)

        if mode == 'default':
            composed.set_assume("({}) | {}".format(assumption_str1, assumption_str2))
            composed.set_guaran(sat_guarantee_str)
            composed.set_sat_guaran(sat_guarantee_str)
            composed.isSat = True
        else:
            composed.set_assume(assumption_str1)   
            composed.set_guaran(guarantee_str)
            composed.isSat = False
        
        return composed

def quotient(c, c2):
    """ Returns the quotient c/c2.

    :param c: A contract c
    :type c: :class:`pycasse.contracts.contract.contract`
    :param c2:  contract c2
    :type c2: :class:`pycasse.contracts.contract.contract`
    :return: A quotient contract c/c2
    :rtype: :class:`pycasse.contracts.contract.contract`
    """
    # Check saturation of c and c2, saturate them if not saturated
    c.checkSat()
    c2.checkSat()

    # Initialize a quotient contract object
    quotient = deepcopy(c)
    quotient.id = (c.id + '/' + c2.id)

    # Merge variables
    merge_variables(quotient, c2)

    # Find assumptions and guarantees for the quotient
    assumption_str1 = quotient.assumption_str
    assumption_str2 = deepcopy(c2.assumption_str)

    guarantee_str1 = quotient.sat_guarantee_str
    guarantee_str2 = deepcopy(c2.sat_guarantee_str)

    quotient.set_assume("({}) & ({})".format(assumption_str1, guarantee_str2))
    quotient.set_guaran("(({}) & ({})) | (!({}))".format(guarantee_str1, assumption_str2, quotient.assumption_str))
    quotient.set_sat_guaran(quotient.guarantee_str)
    quotient.isSat = True

    return quotient

def separation(c, c2):
    """ Returns the separation c%c2

    :param c: A contract c
    :type c: :class:`pycasse.contracts.contract`
    :param c2: A contract c2
    :type c2: :class:`pycasse.contracts.contract`
    :return: A separated contract c%c2
    :rtype: :class:`pycasse.contracts.contract`
    """
    # Check saturation of c and c2, saturate them if not saturated
    c.checkSat()
    c2.checkSat()

    # Initialize a separation contract object
    separation = deepcopy(c)
    separation.id = (c.id + '/' + c2.id)

    # Merge variables
    merge_variables(separation, c2)

    # Find assumptions and guarantees for the quotient
    assumption_str1 = separation.assumption_str
    assumption_str2 = deepcopy(c2.assumption_str)

    guarantee_str1 = separation.sat_guarantee_str
    guarantee_str2 = deepcopy(c2.sat_guarantee_str)

    separation.set_guaran("({}) & ({})".format(guarantee_str1, assumption_str2))
    separation.set_assume("(({}) & ({})) | (!({}))".format(assumption_str1, guarantee_str2, separation.guarantee_str))
    separation.set_sat_guaran(separation.guarantee_str)
    separation.isSat = True

    return separation

def merge_variables(c1, c2):
    # Merge deterministic variables
    for var in c1.deter_var_list:
        if var in c2.deter_var_list:
            # Find index of the variable in different contracts
            idx1 = c1.deter_var_list.index(var)
            idx2 = c2.deter_var_list.index(var)

            # Assert that the variable type is the same
            assert(c1.deter_var_types[idx1] == c2.deter_var_types[idx2])
            
            # Find new bound
            lower_b = max(c1.deter_var_bounds[idx1][0], c2.deter_var_bounds[idx2][0])
            upper_b = min(c1.deter_var_bounds[idx1][1], c2.deter_var_bounds[idx2][1])
            assert(lower_b <= upper_b)
            c1.deter_var_bounds[idx1] = [lower_b, upper_b]

    for var in set(c2.deter_var_list) - set(c1.deter_var_list):
        # Find index of the variable
        idx = c2.deter_var_list.index(var)

        # Append the information of the variable
        c1.deter_var_list.append(var)
        c1.deter_var_types.append(c2.deter_var_types[idx])
        c1.deter_var_bounds.append(c2.deter_var_bounds[idx])

    # Merge parametric variables
    for var in c1.param_var_list:
        if var in c2.param_var_list:
            # Find index of the variable in different contracts
            idx1 = c1.param_var_list.index(var)
            idx2 = c2.param_var_list.index(var)

            # Assert that the variable type is the same
            assert(c1.param_var_types[idx1] == c2.param_var_types[idx2])
            
            # Find new bound
            lower_b = max(c1.param_var_bounds[idx1][0], c2.param_var_bounds[idx2][0])
            upper_b = min(c1.param_var_bounds[idx1][1], c2.param_var_bounds[idx2][1])
            assert(lower_b <= upper_b)
            c1.param_var_bounds[idx1] = [lower_b, upper_b]

    new_params = list(set(c2.param_var_list) - set(c1.param_var_list))
    new_params.sort()
    for var in new_params:
        # Find index of the variable
        idx = c2.param_var_list.index(var)

        # Append the information of the variable
        c1.param_var_list.append(var)
        c1.param_var_types.append(c2.param_var_types[idx])
        c1.param_var_bounds.append(c2.param_var_bounds[idx])

    # Merge non-deterministic variables
    # TODO: may need several modifications if non-deterministic variables are correlated
    curr_nondeter_num = len(c1.nondeter_var_list)
    new_nondeter_num = len(list(set(c2.nondeter_var_list) - set(c1.nondeter_var_list)))
    if curr_nondeter_num == 0:
        c1.nondeter_var_list = c2.nondeter_var_list
        c1.nondeter_var_types = c2.nondeter_var_types
        c1.nondeter_var_mean = c2.nondeter_var_mean
        c1.nondeter_var_cov = c2.nondeter_var_cov
    else:
        c1.nondeter_var_cov = [row + [0]*new_nondeter_num for row in c1.nondeter_var_cov]
        for i, var in enumerate(set(c2.nondeter_var_list) - set(c1.nondeter_var_list)):
            # Find index of the variable
            idx = c2.nondeter_var_list.index(var)

            # Append the information of the variable
            c1.nondeter_var_list.append(var)
            c1.nondeter_var_types.append(c2.nondeter_var_types[idx])
            c1.nondeter_var_mean.append(c2.nondeter_var_mean[idx])
            deque_list = deque(c2.nondeter_var_cov[idx])
            deque_list.rotate(-idx+i)
            c1.nondeter_var_cov += [[0]*curr_nondeter_num + list(deque_list)]