import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

from copy import deepcopy
from gurobipy import GRB
from pystl.variable import DeterVar, NondeterVar, M, EPS
from pystl.vector import Vector, Next
from pystl.core import MILPSolver
from pystl.parser import Parser, ASTObject

M = 10**4
EPS = 10**-4
parser = Parser()

class contract:
    """
    A contract class for defining a contract object.

    :param id: An id of the contract
    :type id: str
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
        self.assumption           = parser('True')
        self.guarantee_str        = 'False'
        self.guarantee            = parser('False')
        self.sat_guarantee_str    = 'False'
        self.sat_guarantee        = parser('False')
        self.isSat                = False
        self.objectives           = []

    def add_deter_vars(self, var_names, dtypes = None, bounds = None):
        """
        Adds controlled variables and their information to the contract 

        :param var_names: A list of names for controlled variables
        :type var_names: list
        :param dtypes: A list of variable types for controlled variables, each entry can be either "BINARY", "INTEGER", or "CONTINUOUS", defaults to "CONTINUOUS", defaults to None
        :type dtypes: list, optional
        :param bounds: An numpy array of lower and upper bounds for controlled variables, defaults to `[-10^4,10^4]` for "CONTINUOUS" and "INTEGER" variable and `[0,1]` for "BINARY", defaults to None
        :type bounds: :class:`numpy.ndarray`, optional
        """
        # For all variables, construct a variable class
        for i, name in enumerate(var_names):
            self.deter_var_list.append(name)
            self.deter_var_types.append(dtypes[i] if dtypes is not None else 'CONTINUOUS')
            self.deter_var_bounds.append(bounds[i] if bounds is not None else [-M, M])

    def add_nondeter_vars(self, var_names, mean, cov, dtypes = None):
        """
        Adds uncontrolled variables and their information to the contract. We plan to add more distribution such as `UNIFORM`, `TRUNCATED_GAUSSIAN`, or even a distribution from `DATA`

        :param var_names: A list of names for uncontrolled variables
        :type var_names: list
        :param mean: A mean vector of uncontrolled variables
        :type mean: :class:`numpy.ndarray`
        :param cov: A covariance matrix of uncontrolled variables
        :type cov: :class:`numpy.ndarray`
        :param dtype: A distribution type for uncontrolled variables, can only be `GAUSSIAN` for now, defaults to None
        :type dtype: str, optional
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
        :param dtypes       : A list of variable types for controlled variables, each entry can be either "BINARY", "INTEGER", or "CONTINUOUS", defaults to "CONTINUOUS"
        :type  dtypes       : list, optional
        :param bounds       : An numpy array of lower and upper bounds for controlled variables, defaults to `[-10^4,10^4]` for "CONTINUOUS" and "INTEGER" variable and `[0,1]` for "BINARY"
        :type  bounds       : :class:`numpy.ndarray`, optional
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
        self.assumption_str = assumption
        self.assumption = parser(assumption)[0][0]
        for variable in self.assumption.variables:
            if variable != 1 and not (variable in self.deter_var_list or variable in self.nondeter_var_list or variable in self.param_var_list):
                raise ValueError("Variable {} not in the contract variables or the dynamics".format(variable))

    def set_guaran(self, guarantee):
        """
        Sets the guarantee of the contract.

        :param guarantee: An STL or StSTL formula which characterizes the guarantee set of the contract
        :type guarantee: str
        """
        self.guarantee_str = guarantee
        self.guarantee = parser(guarantee)[0][0]
        for variable in self.guarantee.variables:
            if variable != 1 and not (variable in self.deter_var_list or variable in self.nondeter_var_list or variable in self.param_var_list):
                raise ValueError("Variable {} not in the contract variables or the dynamics".format(variable))

    def checkSat(self):
        """ Saturates the contract. """
        if self.assumption_str == 'True':
            self.sat_guarantee_str = self.guarantee_str
        elif self.assumption_str == 'False':
            self.sat_guarantee_str = 'True'
        else:
            self.sat_guarantee_str = '({}) -> ({})'.format(self.assumption_str, self.guarantee_str)
        self.sat_guarantee = parser(self.sat_guarantee_str)[0][0]
        self.isSat = True
    
    def checkCompat(self, print_sol=False, verbose = True):
        """ Checks compatibility of the contract. """
        # Build a MILP Solver
        print("====================================================================================")
        print("Checking compatibility of the contract {}...".format(self.id))
        solver = MILPSolver()

        # Add the contract and assumption constraints to the solver
        self.checkSat()
        solver.add_contract(self)
        solver.add_constraint(self.assumption)

        # Solve the problem
        solved = solver.solve()

        # Print the solution
        if verbose and solved:
            print("Contract {} is compatible.\n".format(self.id))
            if print_sol:
                print("Printing a behavior that satisfies the assumptions of the contract {}...".format(self.id))
                solver.print_solution()
        elif verbose and not solved:
            print("Contract {} is not compatible.\n".format(self.id))

        return solved
    
    def checkConsis(self, print_sol=False, verbose = True):
        """ Checks consistency of the contract """
        # Build a MILP Solver
        print("====================================================================================")
        print("Checking consistency of the contract {}...".format(self.id))
        solver = MILPSolver()
        #  solver = SMCSolver()

        # Add the contract and guarantee constraints to the solver
        self.checkSat()
        solver.add_contract(self)
        solver.add_constraint(self.sat_guarantee)

        # Solve the problem
        solved = solver.solve()

        # Print the solution
        if verbose and solved:
            print("Contract {} is consistent.\n".format(self.id))
            if print_sol:
                print("Printing a behavior that satisfies the saturated guarantees of the contract {}...".format(self.id))
                solver.print_solution()
        elif verbose and not solved:
            print("Contract {} is not consistent.\n".format(self.id))

        return solved
    
    def checkFeas(self, print_sol=False, verbose = True):
        """ Checks feasibility of the contract """
        # Build a MILP Solver
        print("====================================================================================")
        print("Checking feasibility of the contract {}...".format(self.id))
        solver = MILPSolver()
        #  solver = SMCSolver()

        # Add the contract and assumption constraints to the solver
        self.checkSat()
        solver.add_contract(self)
        solver.add_constraint(parser("({}) & ({})".format(self.assumption_str, self.guarantee_str))[0][0])

        # Solve the problem
        solved = solver.solve()

        # Print the solution
        if verbose and solved:
            print("Contract {} is feasible.\n".format(self.id))
            if print_sol:
                print("Printing a behavior that satisfies both the assumptions and guarantees of the contract {}...".format(self.id))
                solver.print_solution()
        elif verbose and not solved:
            print("Contract {} is not feasible.\n".format(self.id))

        return solved
    
    def checkRefine(self, contract2refine, print_sol=False):
        """ Checks whether the self contract refines the contract contract2refine"""
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
        #  solver = SMCSolver()

        # Add constraints for refinement condition for assumptions
        print("Checking assumptions condition for refinement...")
        self.checkSat()
        solver.add_contract(self)
        solver.add_constraint(refinement_contract.assumption)
    
        # Check refinement condition for assumptions
        solved = solver.solve()
    
        # Print the counterexample
        if solved:
            print("Assumptions condition for refinement violated. Contract {} does not refine contract {}.\n".format(self.id, contract2refine.id))
            if print_sol:
                print("Printing a counterexample which violates assumptions condition for refinement...")
                solver.print_solution()
            return

        # Resets a MILP Solver
        solver.reset()

        # Add constraints for refinement condition for guarantees
        print("Checking guarantees condition for refinement...")
        self.checkSat()
        solver.add_contract(contract2refine)
        solver.add_constraint(refinement_contract.guarantee)

        # Check refinement condition for guarantees
        solved = solver.solve()

        # Print the counterexample
        if solved:
            print("Guarantees condition for refinement violated. Contract {} does not refine contract {}.\n".format(self.id, contract2refine.id))
            if print_sol:
                print("Printing a counterexample which violates guarantees condition for refinement...")
                solver.print_solution()
            return

        print("Contract {} refines {}.\n".format(self.id, contract2refine.id))
    
    def find_opt_refine_param(self, contract2refine, weights, N=100):
        # Merge Contracts
        c1 = deepcopy(self)
        c1.isSat = False
        (deter_id_map, nondeter_id_map) = c1.merge_contract_variables(contract2refine)

        # Build a MILP Solver
        print("Finding parameters such that contract {} refines contract {}...".format(self.id, contract2refine.id))
        solver = MILPSolver()

        # Add constraints for refinement condition for assumptions
        solver.add_contract(c1)
        assumption1 = deepcopy(c1.assumption)
        assumption2 = deepcopy(contract2refine.assumption)
        assumption2.transform(deter_id_map, nondeter_id_map)
        c1.set_assume(assumption2.implies(assumption1))

        # Add constraints for refinement condition for guarantees
        guarantee1 = deepcopy(c1.sat_guarantee)
        guarantee2 = deepcopy(contract2refine.sat_guarantee)
        guarantee2.transform(deter_id_map, nondeter_id_map)
        c1.set_guaran(guarantee1.implies(guarantee2))
        c1.checkSat()
        # c1.printInfo()

        # Find an optimal set of parameters
        c1.find_opt_param(weights, N=N)
        
    def find_opt_param(self, weights, N=100):
        """ Find an optimal set of parameters for a contract given an objective function. """
        print("Finding an optimal set of parameters for contract {}...".format(self.id))

        # Initialize the initial bounds, SAT, UNSAT, and UNDETERMINED sets
        bounds = deepcopy(self.param_var_bounds)
        param_SAT = []
        param_UNSAT = []
        param_UNDET = [deepcopy(self.param_var_bounds)]

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

        def findPartitionType(partition):
            print(partition)
            # Update the bounds 
            self.param_var_bounds = partition
            # self.printInfo()

            # Build a Solver
            MILPsolver = MILPSolver(mode="Quantitative")
            MILPsolver.add_contract(self)
            MILPsolver.add_constraint(self.assumption, name='b_a')
            MILPsolver.add_constraint(self.guarantee, hard=False, name='b_g')
            # MILPsolver.add_dynamics(sys_dyn)

            # Solve the problem
            MILPsolver.set_objective(sense='minimize')
            if not MILPsolver.solve():
                print("SAT partition!")
                return 0
            else:
                # for v in MILPsolver.model.getVars():
                #     print('%s %g' % (v.varName, v.x))
                    
                MILPsolver.set_objective(sense='maximize')
                if not MILPsolver.solve():
                    print("UNSAT partition!")
                    return 1
                else: 
                    print("UNDET partition!")
                    # for v in MILPsolver.model.getVars():
                    #     print('%s %g' % (v.varName, v.x))
                    return 2

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
        while len(param_UNDET) != 0 and len(param_SAT)+len(param_UNSAT)+len(param_UNDET) <= N:
            tmp_partition = param_UNDET.pop(0)
            partition_type = findPartitionType(tmp_partition)
            if partition_type == 0: # SAT partition
                param_SAT.append(tmp_partition)
            elif partition_type == 1: # UNSAT partition
                param_UNSAT.append(tmp_partition)
            elif partition_type == 2: # UNDET partition
                for partitioned_partition in paramSpacePartition(tmp_partition):
                    param_UNDET.append(partitioned_partition)

        # Double-check UNDET partitions
        tmp_param_UNDET = deepcopy(param_UNDET)
        param_UNDET = []
        for tmp_partition in tmp_param_UNDET:      
            partition_type = findPartitionType(tmp_partition)
            if partition_type == 0: # SAT partition
                param_SAT.append(tmp_partition)
            elif partition_type == 1: # UNSAT partition
                param_UNSAT.append(tmp_partition)
            elif partition_type == 2: # UNDET partition
                param_UNDET.append(tmp_partition)

        # print(param_SAT)
        # print(param_UNSAT)
        # print(param_UNDET)

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
            for weight, variable in zip(weights, variables):
                obj += weight*variable
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
        for partition in param_SAT:
            tmp_minCost, tmp_optimal_params = findMinCost(weights, partition)
            if tmp_minCost < minCost:
                minCost = tmp_minCost
                optimal_params = tmp_optimal_params
        
        print(optimal_params)

        # Plot SAT, UNSAT, and UNDET regions, if the parameter space is 2D
        # if len(self.param_var_list) == 2:
        _, ax = plt.subplots()
        plt.xlabel(self.param_var_list[0])
        plt.ylabel(self.param_var_list[1])
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
        plt.savefig('{}_param_opt.jpg'.format(self.id))
        
        # return optimal_params
        return True

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

def conjunction(c1, c2):
    """ Returns the conjunction of two contracts

    :param c1: A contract c1
    :type c1: :class:`pystl.contracts.contract.contract`
    :param c2: A contract c2
    :type c2: :class:`pystl.contracts.contract.contract`
    :return: A conjoined contract c1^c2
    :rtype: :class:`pystl.contracts.contract.contract`
    """
    # Check saturation of c1 and c2, saturate them if not saturated
    c1.checkSat()
    c2.checkSat()

    # Initialize a conjoined contract object
    conjoined = deepcopy(c1)
    conjoined.id = (c1.id + '^' + c2.id)

    # Merge variables
    merge_variables(conjoined, c2)

    # Find conjoined assumption and guarantee
    conjoined.set_assume("({}) | ({})".format(conjoined.assumption_str, c2.assumption_str))
    conjoined.set_guaran("({}) & ({})".format(conjoined.sat_guarantee_str, c2.sat_guarantee_str))
    conjoined.checkSat()

    return conjoined

def composition(c1, c2):
    """ Returns the composition of two contracts

    :param c1: A contract c1
    :type c1: :class:`pystl.contracts.contract.contract`
    :param c2: A contract c2
    :type c2: :class:`pystl.contracts.contract.contract`
    :return: A composed contract c1*c2
    :rtype: :class:`pystl.contracts.contract.contract`
    """
    # Check saturation of c1 and c2, saturate them if not saturated
    c1.checkSat()
    c2.checkSat()

    # Initialize a conposed contract object
    composed = deepcopy(c1)
    composed.id = (c1.id + '*' + c2.id)

    # Merge variables
    merge_variables(composed, c2)

    # Find composed assumption and guarantee
    composed.set_assume("(({}) & ({})) | (!({})) | (!({}))".format(composed.assumption_str, c2.assumption_str, composed.sat_guarantee_str, c2.sat_guarantee_str))
    composed.set_guaran("({}) & ({})".format(composed.sat_guarantee_str, c2.sat_guarantee_str))
    composed.checkSat()

    return composed

# TODO: quotient not implemented correctly
def quotient(c, c2):
    """ Returns the quotient c/c2

    :param c: A contract c
    :type c: :class:`pystl.contracts.contract.contract`
    :param c2: A contract c2
    :type c2: :class:`pystl.contracts.contract.contract`
    :return: A quotient contract c/c2
    :rtype: :class:`pystl.contracts.contract.contract`
    """
    # Check saturation of c and c2, saturate them if not saturated
    c.checkSat()
    c2.checkSat()

    # Initialize a conposed contract object
    quotient = deepcopy(c)
    quotient.id = (c.id + '/' + c2.id)

    # Merge controlled and uncontrolled variables
    (deter_id_map, nondeter_id_map) = quotient.merge_contract_variables(c2)

    # Find conjoined guarantee, G': G1 and G2
    assumption1 = quotient.assumption
    assumption2 = deepcopy(c2.assumption)
    assumption2.transform(deter_id_map, nondeter_id_map)

    guarantee1 = quotient.sat_guarantee
    guarantee2 = deepcopy(c2.sat_guarantee)
    guarantee2.transform(deter_id_map, nondeter_id_map)

    quotient.assumption = deepcopy(assumption1 & guarantee2)
    quotient.guarantee = deepcopy((guarantee1 & assumption2) | ~assumption1)
    quotient.sat_guarantee = deepcopy(quotient.guarantee)
    quotient.isSat = True

    return quotient

# TODO: separation not implemented correctly
def separation(c, c2):
    """ Returns the separation c%c2

    :param c: A contract c
    :type c: :class:`pystl.contracts.contract.contract`
    :param c2: A contract c2
    :type c2: :class:`pystl.contracts.contract.contract`
    :return: A separated contract c%c2
    :rtype: :class:`pystl.contracts.contract.contract`
    """
    # Check saturation of c1 and c2, saturate them if not saturated
    c.checkSat()
    c2.checkSat()

    # Initialize a conposed contract object
    separation = deepcopy(c)
    separation.id = (c.id + '%' + c2.id)

    # Merge controlled and uncontrolled variables
    (deter_id_map, nondeter_id_map) = separation.merge_contract_variables(c2)

    # Find conjoined guarantee, G': G1 and G2
    assumption1 = separation.assumption
    assumption2 = deepcopy(c2.assumption)
    assumption2.transform(deter_id_map, nondeter_id_map)

    guarantee1 = separation.sat_guarantee
    guarantee2 = deepcopy(c2.sat_guarantee)
    guarantee2.transform(deter_id_map, nondeter_id_map)

    separation.assumption = deepcopy(assumption1 & guarantee2)
    separation.guarantee = deepcopy((guarantee1 & assumption2) | ~assumption1)
    separation.sat_guarantee = deepcopy(separation.guarantee)
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

    for var in set(c2.param_var_list) - set(c1.param_var_list):
        # Find index of the variable
        idx = c2.param_var_list.index(var)

        # Append the information of the variable
        c1.param_var_list.append(var)
        c1.param_var_types.append(c2.param_var_types[idx])
        c1.param_var_bounds.append(c2.param_var_bounds[idx])

    # Merge non-deterministic variables
    curr_nondeter_num = len(c1.nondeter_var_list)
    new_nondeter_num = len(list(set(c2.nondeter_var_list) - set(c1.nondeter_var_list)))
    for var in set(c2.nondeter_var_list) - set(c1.nondeter_var_list):
        # Find index of the variable
        idx = c2.nondeter_var_list.index(var)

        # Append the information of the variable
        c1.nondeter_var_list.append(var)
        c1.nondeter_var_types.append(c2.nondeter_var_types[idx])
        c1.nondeter_var_mean.append(c2.nondeter_var_mean[idx])
        c1.nondeter_var_cov = [row + [0]*new_nondeter_num for row in c1.nondeter_var_cov]
        c1.nondeter_var_cov += [[0]*curr_nondeter_num  + row for row in c2.nondeter_var_cov]