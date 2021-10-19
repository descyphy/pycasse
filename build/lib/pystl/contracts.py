import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

    __slots__ = ('id', 'deter_var_list', 'deter_var_types', 'deter_var_bounds', 'nondeter_var_list', 'nondeter_var_mean', 'nondeter_var_cov', 'param_var_list', 'param_var_types', 'param_var_bounds', 'assumption_str', 'assumption', 'guarantee_str', 'guarantee', 'sat_guarantee_str', 'sat_guarantee', 'isSat', 'objectives')

    def __init__(self, id = ''):
        """ Constructor method """
        self.id = id
        self.deter_var_list       = []
        self.deter_var_types      = []
        self.deter_var_bounds     = []
        self.param_var_list       = []
        self.param_var_types      = []
        self.param_var_bounds     = []
        self.nondeter_var_list    = []
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

    def reset(self):
        """ Resets the contract """
        self.deter_var_list       = []
        self.deter_var_types      = []
        self.deter_var_bounds     = []
        self.param_var_list       = []
        self.param_var_types      = []
        self.param_var_bounds     = []
        self.nondeter_var_list    = []
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

        # Set the mean and covariance matrix
        def convert2var(mat, cov=False):
            parser = Parser(self)
            if cov:
                col = len(mat[0])
                row = len(mat)
                tmp_mat = [[0]*col for i in range(row)]
                for i in range(col):
                    for j in range(row):
                        if isinstance(mat[i][j], str):
                            tmp_mat[i][j] = parser(mat[i][j])[0]
                        else:
                            tmp_mat[i][j] = mat[i][j]
            else:
                col = len(mat)
                tmp_mat = [0]*col
                for i in range(col):
                    if isinstance(mat[i], str):
                        tmp_mat[i] = parser(mat[i])[0]
                    else:
                        tmp_mat[i] = mat[i]
            return tmp_mat

        self.nondeter_var_mean = convert2var(mean)
        self.nondeter_var_cov = convert2var(cov, cov=True)

        # Initialize the variable list
        res = []

        # For all variables, construct a variable class
        for i, name in enumerate(var_names):
            data = NondeterVar(name, len(self.nondeter_var_list), data_type = dtypes[i] if dtypes != None else 'GAUSSIAN')
            self.nondeter_var_list.append(data)
            self.nondeter_var_name2id[name] = len(self.nondeter_var_name2id)
            res.append(data)
        return res

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
        # Initialize the variable list
        res = []

        # For all variables, construct a variable class
        for i, name in enumerate(param_names):
            data = DeterVar(name, len(self.deter_var_list), "parameter", data_type = dtypes[i] if dtypes is not None else 'CONTINUOUS', bound = bounds[i] if bounds is not None else None)
            self.deter_var_list.append(data)
            self.deter_var_name2id[name] = len(self.deter_var_name2id)
            res.append(data)
        return res

    def set_assume(self, assumption):
        """
        Sets the assumption of the contract.

        :param assumption: An STL or StSTL formula which characterizes the assumption set of the contract
        :type assumption: str
        """
        self.assumption_str = assumption
        self.assumption = parser(assumption)[0][0]
        for variable in self.assumption.variables:
            if variable != 1 and not (variable in self.deter_var_list or variable in self.nondeter_var_list):
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
            if variable != 1 and not (variable in self.deter_var_list or variable in self.nondeter_var_list):
                raise ValueError("Variable {} not in the contract variables or the dynamics".format(variable))

    def checkSat(self):
        """ Saturates the contract. """
        if self.assumption_str == 'True':
            self.sat_guarantee_str = self.guarantee_str
        elif self.assumption_str == 'False':
            self.sat_guarantee_str = 'True'
        else:
            self.sat_guarantee_str = '({}) -> ({})'.format(self.assumption_str, self.guarantee_str)
        self.sat_guarantee = parser(self.sat_guarantee_str)[0]
        self.isSat = True
    
    def checkCompat(self, print_sol=False, verbose = True):
        """ Checks compatibility of the contract. """
        # Build a MILP Solver
        if verbose:
            print("Checking compatibility of the contract {}...".format(self.id))
        solver = MILPSolver()

        # Add the contract and assumption constraints to the solver
        solver.add_contract_variables(self)
        solver.add_constraint(self.assumption)

        # Solve the problem
        solved = solver.solve()

        # # Print the solution
        # if verbose and solved:
        #     print("Contract {} is compatible.\n".format(self.id))
        #     if print_sol:
        #         print("Printing a behavior that satisfies the assumptions of the contract {}...".format(self.id))
        #         solver.print_solution()
        # elif verbose and not solved:
        #     print("Contract {} is not compatible.\n".format(self.id))

        # return solved
    
    def checkConsis(self, print_sol=False, verbose = True):
        """ Checks consistency of the contract """
        # Build a MILP Solver
        if verbose:
            print("Checking consistency of the contract {}...".format(self.id))
        solver = MILPSolver(mode='Quantitative')
        #  solver = SMCSolver()

        # Add the contract and assumption constraints to the solver
        self.checkSat()
        solver.add_contract(self)
        solver.add_hard_constraint(self.sat_guarantee)

        # Solve the problem
        solver.preprocess()
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
        if verbose:
            print("Checking feasibility of the contract {}...".format(self.id))
        solver = MILPSolver()
        #  solver = SMCSolver()

        # Add the contract and assumption constraints to the solver
        self.checkSat()
        solver.add_contract(self)
        solver.add_hard_constraint(deepcopy(self.assumption & self.guarantee))

        # Solve the problem
        solver.preprocess()
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
        # Merge Contracts
        c1 = deepcopy(self)
        (deter_id_map, nondeter_id_map) = c1.merge_contract_variables(contract2refine)

        # Build a MILP Solver
        print("Checking whether contract {} refines contract {}...".format(self.id, contract2refine.id))
        solver = MILPSolver()
        #  solver = SMCSolver()

        # Add constraints for refinement condition for assumptions
        print("Checking assumptions condition for refinement...")
        solver.add_contract(c1)
        assumption1 = deepcopy(c1.assumption)
        assumption2 = deepcopy(contract2refine.assumption)
        assumption2.transform(deter_id_map, nondeter_id_map)
        solver.add_hard_constraint(~(assumption2.implies(assumption1)))
    
        # Check refinement condition for assumptions
        solver.preprocess()
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
        solver.add_contract(c1)

        # Add constraints for refinement condition for guarantees
        print("Checking guarantees condition for refinement...")
        guarantee1 = deepcopy(c1.sat_guarantee)
        guarantee2 = deepcopy(contract2refine.sat_guarantee)
        guarantee2.transform(deter_id_map, nondeter_id_map)
        solver.add_hard_constraint(~(guarantee1.implies(guarantee2)))

        # Check refinement condition for guarantees
        solver.preprocess()
        solved = solver.solve()

        # Print the counterexample
        if solved:
            print("Guarantees condition for refinement violated. Contract {} does not refine contract {}.\n".format(self.id, contract2refine.id))
            if print_sol:
                print("Printing a counterexample which violates guarantees condition for refinement...")
                solver.print_solution()
            return

        print("Contract {} refines {}.\n".format(self.id, contract2refine.id))
    
    def merge_contract_variables(self, contract):
        """ Merges contract variables. """
        # Determinate variables
        deter_id_map = []
        for var in contract.deter_var_list:
            if var.name in self.deter_var_name2id:
                deter_id_map.append(self.deter_var_name2id[var.name])
            else:
                self.deter_var_list.append(var)
                var_len = len(self.deter_var_name2id)
                self.deter_var_name2id[var.name] = var_len
                deter_id_map.append(var_len)
        #  print(deter_id_map)

        # Nondeterminate variables
        nondeter_id_map = []
        extra_nondeter_id = []
        for i, var in enumerate(contract.nondeter_var_list):
            if var.name in self.nondeter_var_name2id:
                nondeter_id_map.append(self.nondeter_var_name2id[var.name])
            else:
                self.nondeter_var_list.append(var)
                var_len = len(self.nondeter_var_name2id)
                self.nondeter_var_name2id[var.name] = var_len
                nondeter_id_map.append(var_len)
                extra_nondeter_id.append(i)
        #  print(nondeter_id_map)

        # print(extra_nondeter_id)
        # extra_nondeter_id = np.array(extra_nondeter_id)
        self_len = len(self.nondeter_var_cov)
        contract_len = len(extra_nondeter_id)
        if contract_len > 0:
            if self_len == 0:
                self.nondeter_var_mean = contract.nondeter_var_mean[extra_nondeter_id]
                self.nondeter_var_cov = contract.nondeter_var_cov[extra_nondeter_id, extra_nondeter_id]
            else:
                # print(self.nondeter_var_mean)
                # print(contract.nondeter_var_mean)
                # print(extra_nondeter_id)
                for id in extra_nondeter_id:
                    self.nondeter_var_mean += [contract.nondeter_var_mean[id]]

                for row in self.nondeter_var_cov:
                    row += [0]*contract_len
                for id in extra_nondeter_id:
                    print([0]*self_len + contract.nondeter_var_cov[id])
                    self.nondeter_var_cov.append([0]*self_len + contract.nondeter_var_cov[id])
                
                # print(self.nondeter_var_mean)
                # print(self.nondeter_var_cov)

        return (np.array(deter_id_map), np.array(nondeter_id_map))
    
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
        c1.printInfo()

        # Find an optimal set of parameters
        c1.find_opt_param(weights, N=N)
        
    def find_opt_param(self, weights, N=100):
        """ Find an optimal set of parameters for a contract given an objective function. """
        
        print("Finding an optimal set of parameters for contract {}...".format(self.id))
        
        # Find the parameters and their bounds
        variable = []
        var_info = {}
        bounds = []
        param_count = 0
        for v in self.deter_var_list:
            if v.var_type == 'parameter' and ('p' in v.name or 'c' in v.name):
                variable.append(True)
                var_info[str(v.name)] = [self.deter_var_name2id[str(v.name)], param_count]
                param_count += 1
                bounds.append(v.bound)
            else:
                variable.append(False)
        variable = np.array(variable)
        # print(variable)
        # print(var_info)
        # print(bounds)
        # input()

        # Initialize the SAT, UNSAT, and UNDETERMINED sets
        param_SAT = []
        param_UNSAT = []
        param_UNDET = [deepcopy(bounds)]

        # Pre-partition the parameter space for probability thresholds
        count = 0
        for var_name in var_info.keys():
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

        def findPartitionType(var_info, partition):
            print(partition)

            for var in self.deter_var_list:
                if var.name in list(var_info.keys()):
                    var.bound = np.array(partition[var_info[var.name][1]])
            # print(self.deter_var_list)

            # Build a Solver
            MILPsolver = MILPSolver(mode="Quantitative")
            MILPsolver.add_contract(self)
            MILPsolver.add_hard_constraint(self.assumption)
            MILPsolver.add_soft_constraint(self.guarantee)
            # MILPsolver.add_dynamics(sys_dyn)

            # Solve the problem
            MILPsolver.preprocess()
            MILPsolver.set_objective(sense='minimize')
            if not MILPsolver.solve():
                print("SAT partition!")
                return 0
            else:
                for v in MILPsolver.model.getVars():
                    print('%s %g' % (v.varName, v.x))
                    
                MILPsolver.set_objective(sense='maximize')
                if not MILPsolver.solve():
                    print("UNSAT partition!")
                    return 1
                else: 
                    print("UNDET partition!")
                    for v in MILPsolver.model.getVars():
                        print('%s %g' % (v.varName, v.x))
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
            partition_type = findPartitionType(var_info, tmp_partition)
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
            partition_type = findPartitionType(var_info, tmp_partition)
            if partition_type == 0: # SAT partition
                param_SAT.append(tmp_partition)
            elif partition_type == 1: # UNSAT partition
                param_UNSAT.append(tmp_partition)
            elif partition_type == 2: # UNDET partition
                param_UNDET.append(tmp_partition)

        # print(param_SAT)
        # print(param_UNSAT)
        # print(param_UNDET)

        def findMinCost(var_info, weights, partition):
            # Initialize Gurobi model
            model = gp.Model()
            model.setParam("OutputFlag", 0)
            
            # Add variables
            variables = []
            for var_name, var_idx in var_info.items():
                variables.append(model.addVar(lb=partition[var_idx[1]][0], ub=partition[var_idx[1]][1], vtype=GRB.CONTINUOUS, name=var_name))
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
            for var_name in var_info.keys():
                optimal_params[var_name] = model.getVarByName(var_name).x

            # print(obj.getValue())
            # print(optimal_params)
            
            return obj.getValue(), optimal_params


        # Find the optimal set of parameters
        minCost = 10**4
        optimal_params = {}
        for partition in param_SAT:
            tmp_minCost, tmp_optimal_params = findMinCost(var_info, weights, partition)
            if tmp_minCost < minCost:
                minCost = tmp_minCost
                optimal_params = tmp_optimal_params
        
        print(optimal_params)

        # Plot SAT, UNSAT, and UNDET regions, if the parameter space is 2D
        if len(bounds) == 2:
            _, ax = plt.subplots()
            plt.xlabel(list(var_info.keys())[0])
            plt.ylabel(list(var_info.keys())[1])
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
                res += "    {}, {}, {}\n".format(v, self.nondeter_var_types[i], self.nondeter_var_bounds[i])
            res += "    mean: {}\n".format(self.nondeter_var_mean)
            res += "    cov: {}\n".format(self.nondeter_var_cov)
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

    # Merge controlled and uncontrolled variables
    (deter_id_map, nondeter_id_map) = conjoined.merge_contract_variables(c2)

    # Find conjoined guarantee, G': G1 and G2
    assumption1 = deepcopy(conjoined.assumption)
    assumption2 = deepcopy(c2.assumption)
    assumption2.transform(deter_id_map, nondeter_id_map)
    conjoined.assumption = assumption1 | assumption2

    guarantee1 = deepcopy(conjoined.sat_guarantee)
    guarantee2 = deepcopy(c2.sat_guarantee)
    guarantee2.transform(deter_id_map, nondeter_id_map)

    conjoined.guarantee = guarantee1 & guarantee2
    conjoined.sat_guarantee = deepcopy(conjoined.guarantee)
    conjoined.isSat = True

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

    # Merge controlled and uncontrolled variables
    (deter_id_map, nondeter_id_map) = composed.merge_contract_variables(c2)

    # Find conjoined guarantee, G': G1 and G2
    assumption1 = composed.assumption
    assumption2 = deepcopy(c2.assumption)
    assumption2.transform(deter_id_map, nondeter_id_map)

    guarantee1 = composed.sat_guarantee
    guarantee2 = deepcopy(c2.sat_guarantee)
    guarantee2.transform(deter_id_map, nondeter_id_map)

    composed.assumption = deepcopy((assumption1 & assumption2) | ~guarantee1 | ~guarantee2)
    composed.guarantee = deepcopy(guarantee1 & guarantee2)
    composed.sat_guarantee = deepcopy(composed.guarantee)
    composed.isSat = True

    return composed

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