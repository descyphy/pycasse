import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt

from itertools import combinations
from copy import deepcopy
from numpy.lib.npyio import save
from pystl.variable import DeterVar, NondeterVar, M, EPS
from pystl.vector import Vector, Next
from pystl.parser import P, true, false, And, Or, Globally, Eventually, Until, Release, Parser
from pystl.core import SMCSolver, MILPSolver


class contract:
    """
    A contract class for defining a contract object.

    :param id: An id of the contract
    :type id: str
    """

    __slots__ = ('id', 'deter_var_list', 'deter_var_name2id', 'nondeter_var_list', 'nondeter_var_name2id', 'nondeter_var_mean', 'nondeter_var_cov', 'assumption', 'guarantee', 'sat_guarantee', 'isSat', 'objectives')

    def __init__(self, id = ''):
        """ Constructor method """
        self.id = id
        self.deter_var_list       = [None]
        self.deter_var_name2id    = {None: 0}
        self.nondeter_var_list    = []
        self.nondeter_var_name2id = {}
        self.nondeter_var_mean    = np.empty(0)
        self.nondeter_var_cov     = np.empty(0)
        self.assumption           = true
        self.guarantee            = false
        self.sat_guarantee        = false
        self.isSat                = False
        self.objectives           = []

    def set_controlled_vars(self, var_names, dtypes = None, bounds = None):
        """ 
        Adds controlled variables and their information to the contract 

        :param  var_names   : A list of names for controlled variables
        :type   var_names   : list
        :param  dtypes      : A list of variable types for controlled variables, each entry can be either "BINARY", "INTEGER", or "CONTINUOUS", defaults to "CONTINUOUS"
        :type   dtypes      : list, optional
        :param  bounds      : An numpy array of lower and upper bounds for controlled variables, defaults to `[-10^4,10^4]` for "CONTINUOUS" and "INTEGER" variable and `[0,1]` for "BINARY"
        :type   bounds      : :class:`numpy.ndarray`, optional
        """
        # Initialize the variable list
        res = []

        # For all variables, construct a variable class
        for i, name in enumerate(var_names):
            data = DeterVar(name, len(self.deter_var_list), "controlled", data_type = dtypes[i] if dtypes is not None else 'CONTINUOUS', bound = bounds[i] if bounds is not None else None)
            self.deter_var_list.append(data)
            self.deter_var_name2id[name] = len(self.deter_var_name2id)
            res.append(data)
        return res

    def set_deter_uncontrolled_vars(self, var_names, dtypes = None, bounds = None):
        """
        Adds deterministic uncontrolled variables and their information to the contract

        :param  var_names   : A list of names for uncontrolled variables
        :type   var_names   : list
        :param  dtypes      : A list of variable types for controlled variables, each entry can be either "BINARY", "INTEGER", or "CONTINUOUS", defaults to "CONTINUOUS"
        :type   dtypes      : list, optional
        :param  bounds      : An numpy array of lower and upper bounds for controlled variables, defaults to `[-10^4,10^4]` for "CONTINUOUS" and "INTEGER" variable and `[0,1]` for "BINARY"
        :type   bounds      : :class:`numpy.ndarray`, optional
        """
        # Initialize the variable list
        res = []

        # For all variables, construct a variable class
        for i, name in enumerate(var_names):
            data = DeterVar(name, len(self.deter_var_list), "uncontrolled", data_type = dtypes[i] if dtypes is not None else 'CONTINUOUS', bound = bounds[i] if bounds is not None else None)
            self.deter_var_list.append(data)
            self.deter_var_name2id[name] = len(self.deter_var_name2id)
            res.append(data)
        return res

    def set_nondeter_uncontrolled_vars(self, var_names, mean, cov, dtypes = None):
        """
        Adds uncontrolled variables and their information to the contract. We plan to add more distribution such as `UNIFORM`, `TRUNCATED_GAUSSIAN`, or even a distribution from `DATA`

        :param  var_names: A list of names for uncontrolled variables
        :type   var_names: list
        :param  mean     : A mean vector of uncontrolled variables
        :type   mean     : :class:`numpy.ndarray`
        :param  cov      : A covariance matrix of uncontrolled variables
        :type   cov      : :class:`numpy.ndarray`
        :param  dtype    : A distribution type for uncontrolled variables, can only be `GAUSSIAN` for now, defaults to `GAUSSIAN`
        :type   dtype    : str, optional
        """
        # Check the dimensions of mean and covariance matrix
        assert(mean.shape == (len(var_names),))
        assert(cov.shape == (len(var_names), len(var_names)))

        # Set the mean and covariance matrix
        self.nondeter_var_mean = mean
        self.nondeter_var_cov = cov

        # Initialize the variable list
        res = []

        # For all variables, construct a variable class
        for i, name in enumerate(var_names):
            data = NondeterVar(name, len(self.nondeter_var_list), data_type = dtypes[i] if dtypes != None else 'GAUSSIAN')
            self.nondeter_var_list.append(data)
            self.nondeter_var_name2id[name] = len(self.nondeter_var_name2id)
            res.append(data)
        return res

    def set_params(self, param_names, dtypes = None, bounds = None):
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
        if (isinstance(assumption, str)): # If the assumption is given as a string
            parser = Parser(self) # Parse the string into an AST
            self.assumption = parser(assumption)
        elif (isinstance(assumption, ASTObject)): # If the assumption is given as an AST # TODO: Where is ASTObject defined?
            self.assumption = assumption
        else: assert(False)

    def set_guaran(self, guarantee):
        """
        Sets the guarantee of the contract.

        :param guarantee: An STL or StSTL formula which characterizes the guarantee set of the contract
        :type guarantee: str
        """
        if (isinstance(guarantee, str)): # If the guarantee is given as a string
            parser = Parser(self) # Parse the string into an AST
            self.guarantee = parser(guarantee)
        elif (isinstance(guarantee, ASTObject)): # If the guarantee is given as an AST # TODO: Where is ASTObject defined?
            self.guarantee = guarantee
        else: assert(False)

    def checkSat(self):
        """ Saturates the contract. """
        if not self.isSat:
            self.isSat = True
            assumption = deepcopy(self.assumption)
            guarantee = deepcopy(self.guarantee)
            self.sat_guarantee = assumption.implies(guarantee)
    
    def checkCompat(self, print_sol=False, verbose = True):
        """ Checks compatibility of the contract. """
        # Build a MILP Solver
        if verbose:
            print("Checking compatibility of the contract {}...".format(self.id))
        solver = MILPSolver()
        #  solver = SMCSolver()

        # Add the contract and assumption constraints to the solver
        solver.add_contract(self)
        solver.add_constraint(self.assumption)

        # Solve the problem
        solved = solver.solve()

        # Print the solution
        if verbose and solved:
            print("Contract {} is compatible.\n".format(self.id))
            if print_sol:
                solver.print_solution()
        elif verbose and not solved:
            print("Contract {} is not compatible.\n".format(self.id))

        return solved
    
    def checkConsis(self, print_sol=False, verbose = True):
        """ Checks consistency of the contract """
        # Build a MILP Solver
        if verbose:
            print("Checking consistency of the contract {}...".format(self.id))
        solver = MILPSolver()
        #  solver = SMCSolver()

        # Add the contract and assumption constraints to the solver
        self.checkSat()
        solver.add_contract(self)
        solver.add_constraint(self.sat_guarantee)

        # Solve the problem
        solved = solver.solve()

        # Print the solution
        if verbose and solved:
            print("Contract {} is consistent.\n".format(self.id))
            if print_sol:
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
        solver.add_constraint(self.assumption & self.guarantee)

        # Solve the problem
        solved = solver.solve()

        # Print the solution
        if verbose and solved:
            print("Contract {} is feasible.\n".format(self.id))
            if print_sol:
                print("Printing a behavior that satisfies both the assumption and guarantee of the contract {}...".format(self.id))
                solver.print_solution()
        elif verbose and not solved:
            print("Contract {} is not feasible.\n".format(self.id))

        return solved
    
    def checkRefine(self, contract2refine, print_sol=False):
        """ Checks whether contract2refine refines the contract """
        # Merge Contracts
        c1 = deepcopy(self)
        (deter_id_map, nondeter_id_map) = c1.merge_contract_variables(contract2refine)

        # Build a MILP Solver
        print("Checking whether contract {} refines contract {}...".format(self.id, contract2refine.id))
        solver = MILPSolver()
        #  solver = SMCSolver()

        # Add constraints for refinement condition 1
        print("Checking condition 1 for refinement...")
        solver.add_contract(c1)
        assumption1 = deepcopy(c1.assumption)
        assumption2 = deepcopy(contract2refine.assumption)
        assumption2.transform(deter_id_map, nondeter_id_map)
        solver.add_constraint(~(assumption2.implies(assumption1)))
        
        # Check refinement condition 1
        solved = solver.solve()

        # Print the counterexample
        if solved:
            print("Condition 1 for refinement violated. Contract {} does not refine contract {}.\n".format(self.id, contract2refine.id))
            if print_sol:
                solver.print_solution()
            return

        # Resets a MILP Solver
        solver.reset()
        solver.add_contract(c1)

        # Add constraints for refinement condition 2
        print("Checking condition 2 for refinement...")
        guarantee1 = deepcopy(c1.guarantee)
        guarantee2 = deepcopy(contract2refine.guarantee)
        guarantee2.transform(deter_id_map, nondeter_id_map)
        solver.add_constraint(~(guarantee1.implies(guarantee2)))

        # Check refinement condition 2
        solved = solver.solve()

        # Print the counterexample
        if solved:
            print("Condition 2 for refinement violated. Contract {} does not refine contract {}.\n".format(self.id, contract2refine.id))
            if print_sol:
                print("Printing a counterexample which violates condition 2 for refinement...")
                solver.print_solution()
            return

        print("Contract {} refines {}.\n".format(self.id, contract2refine.id))
    
    def merge_contract_variables(self, contract):
        """ Merges contract variables. """
        # Determinate variables
        deter_id_map = [0]
        for var in contract.deter_var_list[1:]:
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

        # 
        extra_nondeter_id = np.array(extra_nondeter_id)
        self_len = len(self.nondeter_var_cov)
        contract_len = len(extra_nondeter_id)
        if contract_len > 0:
            if self_len == 0:
                self.nondeter_var_mean = contract.nondeter_var_mean[extra_nondeter_id]
                self.nondeter_var_cov = contract.nondeter_var_cov[extra_nondeter_id, extra_nondeter_id]
            else:
                self.nondeter_var_mean = np.concatenate((self.nondeter_var_mean, contract.nondeter_var_mean[extra_nondeter_id]))
                #  print(self.nondeter_var_cov)
                #  print(np.zeros((self_len, contract_len)))
                #  print(np.zeros((contract_len, self_len)))
                #  print(contract.nondeter_var_cov[extra_nondeter_id, extra_nondeter_id])
                self.nondeter_var_cov = np.block([[self.nondeter_var_cov, np.zeros((self_len, contract_len))], [np.zeros((contract_len, self_len)), contract.nondeter_var_cov[extra_nondeter_id, extra_nondeter_id]]])
        return (np.array(deter_id_map), np.array(nondeter_id_map))
    
    # def find_opt_param(self, objective, N=100):
    #     """ Find an optimal set of parameters for a contract given an objective function. """
    #     # Build a MILP Solver
    #     print("Finding an optimal set of parameters for contract {}...".format(self.id))


    #     variable = [False]
    #     bounds = []
    #     for v in self.deter_var_list[1:]:
    #         if v.var_type == 'parameter':
    #             variable.append(True)
    #             bounds.append(v.bound)
    #         else:
    #             variable.append(False)
    #     variable = np.array(variable)
    #     bounds = np.array(bounds)
    #     # Sample the parameters N times
    #     sampled_param = np.random.rand(N, len(bounds))
    #     sampled_param *= (bounds[:,1] - bounds[:,0])
    #     sampled_param += bounds[:,0]
    #     #  print(variable)
    #     #  print(bounds)
    #     #  print(sampled_param)
    #     #  input()
    #     def change(data, variable, values):
    #         #  print(data)
    #         #  print(variable)
    #         #  print(values)
    #         parameter = data[variable[:len(data)]] 
    #         data[0] += np.sum(parameter * values[:len(parameter)])
    #         data[variable[:len(data)]] = 0
    #     def traverse(node, variable, values):
    #         if node.ast_type == 'AP':
    #             change(node.expr.deter_data, variable, values)
    #         elif node.ast_type == 'StAP':
    #             #  print(node.prob)
    #             change(node.expr.deter_data, variable, values)
    #             change(node.prob.deter_data, variable, values)
    #         else:
    #             for f in node.formula_list:
    #                 traverse(f, variable, values)

    #     # Build a deepcopy of the contract
    #     c = deepcopy(self)

    #     x_id = np.argmax(variable)
    #     y_id = x_id + 1 + np.argmax(variable[x_id + 1:])

    #     fig = plt.figure()
    #     plt.xlabel(self.deter_var_list[x_id].name)
    #     plt.ylabel(self.deter_var_list[y_id].name)
    #     plt.xlim(self.deter_var_list[x_id].bound[0], self.deter_var_list[x_id].bound[1])
    #     plt.ylim(self.deter_var_list[y_id].bound[0], self.deter_var_list[y_id].bound[1])

    #     for i in range(N):
    #         #  print(sampled_param[i])
    #         c.assumption = deepcopy(self.assumption)
    #         traverse(c.assumption, variable, sampled_param[i])
    #         c.guarantee = deepcopy(self.guarantee)
    #         traverse(c.guarantee, variable, sampled_param[i])
    #         #  print(c)
    #         #  input()
    #         if c.checkFeas(print_sol = False, verbose = False):
    #             plt.plot(sampled_param[i, 0], sampled_param[i, 1], 'go')
    #         else:
    #             plt.plot(sampled_param[i, 0], sampled_param[i, 1], 'ro')

    #     #  plt.show()
    #     #  plt.savefig('test.jpg')

    def printInfo(self):
        print(str(self))

    def __str__(self):
        """ Prints information of the contract """
        res = ""
        res += "Contract ID: {}\n".format(self.id)
        for v in self.deter_var_list[1:]:
            res += "\n  "
            res += "\n    ".join(str(v).splitlines())
        for v in self.nondeter_var_list:
            res += "\n  "
            res += "\n    ".join(str(v).splitlines())
            res += "    mean: {}\n".format(self.nondeter_var_mean)
            res += "    cov: {}\n".format(self.nondeter_var_cov)
        res += "\n"
        res += "  Assumption: {}\n".format(self.assumption)
        res += "  Guarantee: {}\n".format(self.guarantee)
        res += "  Saturated Guarantee: {}\n".format(self.sat_guarantee)
        res += "  isSat: {}\n".format(self.isSat)
        return res

    def __repr__(self):
        """ Prints information of the contract """
        res = ""
        res += "Contract ID: {}".format(self.id)
        for v in self.deter_var_list[1:]:
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
    conjoined.merge_contract(c2)

    # Find conjoined guarantee, G': G1 and G2
    conjoined.assumption = conjoined.assumption | deepcopy(c2.assumption)
    conjoined.guarantee = conjoined.sat_guarantee & deepcopy(c2.sat_guarantee)
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
    composed.merge_contract(c2)

    # Find conjoined guarantee, G': G1 and G2
    composed.assumption = (composed.assumption & deepcopy(c2.assumption)) | ~deepcopy(composed.sat_guarantee) | ~deepcopy(c2.sat_guarantee)
    composed.guarantee = composed.sat_guarantee & deepcopy(c2.sat_guarantee)
    composed.sat_guarantee = deepcopy(composed.guarantee)
    composed.isSat = True

    return composed

def quotient(c, c2):
    pass
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
    quotient.merge_contract(c2)

    # Find conjoined guarantee, G': G1 and G2
    quotient.assumption = (quotient.assumption & deepcopy(c2.sat_guarantee))
    quotient.guarantee = ~(quotient.sat_guarantee & deepcopy(c2.assumption)) | deepcopy(quotient.assumption)
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
    separation.merge_contract(c2)

    # Find conjoined guarantee, G': G1 and G2
    separation.assumption = (separation.assumption & deepcopy(c2.sat_guarantee))
    separation.guarantee = (separation.sat_guarantee & deepcopy(c2.assumption)) | ~deepcopy(separation.assumption)
    separation.sat_guarantee = deepcopy(separation.guarantee)
    separation.isSat = True

    return separation

def env_load(H, init=None, savepath=True):
    """ Create a set of contracts for the vehicles on the highway environment.

    :param data: [description]
    :type data: [type]
    """
    # TODO: Temporary data goes here
    # highway
    data = {"scenario": "highway", "vehicle": {"id": [0, 1, 2], "width": 2.0, "length": 5.0, "target": [[100.0, 8.0], [100.0, 8.0], [100.0, 12.0]], "region": {"format": [["A", "B", "C", "D", "E"]], "representation": "Ax^2 + Bx + Cy^2 + Dy + E <= 0", "equation": [[[[0, -0.0, 0, 1.0, -10.0], [0, 0.0, 0, -1.0, 6.0], [0, -1.0, 0, -0.0, 0.0], [0, 1.0, 0, 0.0, -10000.0]]], [[[0, -0.0, 0, 1.0, -10.0], [0, 0.0, 0, -1.0, 6.0], [0, -1.0, 0, -0.0, 0.0], [0, 1.0, 0, 0.0, -10000.0]]], [[[0, -0.0, 0, 1.0, -14.0], [0, 0.0, 0, -1.0, 10.0], [0, -1.0, 0, -0.0, 0.0], [0, 1.0, 0, 0.0, -10000.0]]]]}}, "dynamics": {"x": ["d_x", "d_y", "v_x", "v_y"], "u": ["a_x", "a_y"], "dt": 1, "A": [[1, 0, "dt", 0], [0, 1, 0, "dt"], [0, 0, 1, 0], [0, 0, 0, 1]], "B": [[0, 0], [0, 0], ["dt", 0], [0, "dt"]]}, "physics": {"velocity_bound": 30, "acceleration_bound": 10}}
    # # highway_merging
    # data = {"scenario": "intersection", "vehicle": {"id": [0, 1], "width": 2.0, "length": 5.0, "target": [[-114.0, 2.000000000000007], [-2.0000000000000138, -114.0]], "region": {"format": [["A", "B", "C", "D", "E"]], "representation": "Ax^2 + Bx + Cy^2 + Dy + E <= 0", "equation": [[[[0, 1.0, 0, 0.0, 0.0], [0, -1.0, 0, -0.0, -4.0], [0, -0.0, 0, 1.0, -114.0], [0, 0.0, 0, -1.0, 14.0]], [[1, 28.0, 1, -28.0, 196.0], [-1, -28.0, -1, 28.0, -292.0], [0, -0.0, 0, 1.0, -14.0], [0, -1.0, 0, -6.123233995736766e-17, -14.0]], [[0, -6.217248937900877e-17, 0, -1.0, 0.0], [0, 6.217248937900877e-17, 0, 1.0, -4.0], [0, 1.0, 0, -6.217248937900877e-17, 14.0], [0, -1.0, 0, 6.217248937900877e-17, -114.0]]], [[[0, 5.995204332975846e-17, 0, 1.0, 0.0], [0, -5.995204332975846e-17, 0, -1.0, -4.0], [0, -1.0, 0, 5.995204332975846e-17, -114.0], [0, 1.0, 0, -5.995204332975846e-17, 14.0]], [[1, 28.0, 1, 28.0, 196.0], [-1, -28.0, -1, -28.0, -292.0], [0, -1.0, 0, 6.123233995736766e-17, -14.0], [0, 0.0, 0, -1.0, -14.0]], [[0, 1.0, 0, -1.1990408665951691e-16, 0.0], [0, -1.0, 0, 1.1990408665951691e-16, -4.0], [0, 1.1990408665951691e-16, 0, 1.0, 14.0], [0, -1.1990408665951691e-16, 0, -1.0, -114.0]]]]}}, "dynamics": {"x": ["d_x", "d_y", "v_x", "v_y"], "u": ["a_x", "a_y"], "dt": 1, "A": [[1, 0, "dt", 0], [0, 1, 0, "dt"], [0, 0, 1, 0], [0, 0, 0, 1]], "B": [[0, 0], [0, 0], ["dt", 0], [0, "dt"]]}, "physics": {"velocity_bound": 30, "acceleration_bound": 10}}
    # # intersection
    # data = {"scenario": "intersection", "vehicle": {"id": [0, 1], "width": 2.0, "length": 5.0, "target": [[-114.0, 2.000000000000007], [-2.0000000000000138, -114.0]], "region": {"format": [["A", "B", "C", "D", "E"]], "representation": "Ax^2 + Bx + Cy^2 + Dy + E <= 0", "equation": [[[[0, 1.0, 0, 0.0, 0.0], [0, -1.0, 0, -0.0, -4.0], [0, -0.0, 0, 1.0, -114.0], [0, 0.0, 0, -1.0, 14.0]], [[1, 28.0, 1, -28.0, 196.0], [-1, -28.0, -1, 28.0, -292.0], [0, -0.0, 0, 1.0, -14.0], [0, -1.0, 0, -6.123233995736766e-17, -14.0]], [[0, -6.217248937900877e-17, 0, -1.0, 0.0], [0, 6.217248937900877e-17, 0, 1.0, -4.0], [0, 1.0, 0, -6.217248937900877e-17, 14.0], [0, -1.0, 0, 6.217248937900877e-17, -114.0]]], [[[0, 5.995204332975846e-17, 0, 1.0, 0.0], [0, -5.995204332975846e-17, 0, -1.0, -4.0], [0, -1.0, 0, 5.995204332975846e-17, -114.0], [0, 1.0, 0, -5.995204332975846e-17, 14.0]], [[1, 28.0, 1, 28.0, 196.0], [-1, -28.0, -1, -28.0, -292.0], [0, -1.0, 0, 6.123233995736766e-17, -14.0], [0, 0.0, 0, -1.0, -14.0]], [[0, 1.0, 0, -1.1990408665951691e-16, 0.0], [0, -1.0, 0, 1.1990408665951691e-16, -4.0], [0, 1.1990408665951691e-16, 0, 1.0, 14.0], [0, -1.1990408665951691e-16, 0, -1.0, -114.0]]]]}}, "dynamics": {"x": ["d_x", "d_y", "v_x", "v_y"], "u": ["a_x", "a_y"], "dt": 1, "A": [[1, 0, "dt", 0], [0, 1, 0, "dt"], [0, 0, 1, 0], [0, 0, 0, 1]], "B": [[0, 0], [0, 0], ["dt", 0], [0, "dt"]]}, "physics": {"velocity_bound": 30, "acceleration_bound": 10}}
    env = data["scenario"]

    # Find dynamics
    dt = int(data["dynamics"]["dt"])
    A = np.array(data["dynamics"]["A"])
    B = np.array(data["dynamics"]["B"])
    A = np.where(A=="dt", dt, A)
    B = np.where(B=="dt", dt, B)
    A = A.astype(float)
    B = B.astype(float)
    
    if savepath:
        # Find region parameters
        param = np.array(data["vehicle"]["region"]["equation"])

        # Initialize plots and settings
        if env in ("highway", "highway_merging"):
            _, ax = plt.subplots(len(param), 1)
        else:
            _, ax = plt.subplots(1, len(param))
        cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

        # For each vehicle, 
        for vehicle_num in data["vehicle"]["id"]:
            # For each region of a vehicle, find vertices and plot the region
            region_count = 0
            
            im = None
            for region_param in param[vehicle_num]:
                f1 = lambda x,y : region_param[0,0]*x**2 + region_param[0,1]*x + region_param[0,2]*y**2 + region_param[0,3]*y + region_param[0,4]
                f2 = lambda x,y : region_param[1,0]*x**2 + region_param[1,1]*x + region_param[1,2]*y**2 + region_param[1,3]*y + region_param[1,4]
                f3 = lambda x,y : region_param[2,0]*x**2 + region_param[2,1]*x + region_param[2,2]*y**2 + region_param[2,3]*y + region_param[2,4]
                f4 = lambda x,y : region_param[3,0]*x**2 + region_param[3,1]*x + region_param[3,2]*y**2 + region_param[3,3]*y + region_param[3,4]

                if env in ("highway", "highway_merging"):
                    x = np.linspace(-10,550,1000)
                    y = np.linspace(-20,20,1000)
                else:
                    x = np.linspace(-50,50,1000)
                    y = np.linspace(-50,50,1000)
                x,y = np.meshgrid(x,y)

                if im is None:
                    im = ((f1(x,y)<=0) & (f2(x,y)<=0) & (f3(x,y)<=0) & (f4(x,y)<=0)).astype(int) 
                else:
                    im += ((f1(x,y)<=0) & (f2(x,y)<=0) & (f3(x,y)<=0) & (f4(x,y)<=0)).astype(int) 

                ax[vehicle_num].set_title("Vehicle {}".format(vehicle_num), fontsize=10)
                region_count += 1
            ax[vehicle_num].imshow(im, extent=(x.min(),x.max(),y.min(),y.max()), origin="lower", cmap=cmaps[vehicle_num])
                
        plt.savefig("test_{}.png".format(env))

    # Create a contract for each vehicle
    for vehicle_num in data["vehicle"]["id"]:
        # Build a MILP solver
        tmp_solver = MILPSolver()

        # Initialize a contract
        tmp_contract = contract("vehicle_{}".format(vehicle_num))

        # Set deterministic uncontrolled and controlled variables
        velocity_bound = data["physics"]["velocity_bound"]
        acceleration_bound = data["physics"]["acceleration_bound"]
        uncontrolled_vars = []
        controlled_vars = []
        uncontrolled_bounds = np.empty((0,2))
        controlled_bounds = np.empty((0,2))

        for tmp_vehicle_num in data["vehicle"]["id"]:
            x_tmp = [s + "_{}".format(tmp_vehicle_num) for s in data["dynamics"]["x"]]
            u_tmp = [s + "_{}".format(tmp_vehicle_num) for s in data["dynamics"]["u"]]
            uncontrolled_vars = uncontrolled_vars + x_tmp
            uncontrolled_bounds = np.append(uncontrolled_bounds, np.array([[-M, M], [-M, M], [-velocity_bound, velocity_bound], [-velocity_bound, velocity_bound]]), axis=0)
            if tmp_vehicle_num == vehicle_num:
                controlled_vars = controlled_vars + u_tmp
                controlled_bounds = np.append(controlled_bounds, np.array([[-acceleration_bound, acceleration_bound], [-acceleration_bound, acceleration_bound]]), axis=0)
            else:
                uncontrolled_vars = uncontrolled_vars + u_tmp
                uncontrolled_bounds = np.append(uncontrolled_bounds, np.array([[-acceleration_bound, acceleration_bound], [-acceleration_bound, acceleration_bound]]), axis=0)

        uncontrolled_vars = tmp_contract.set_deter_uncontrolled_vars(uncontrolled_vars, bounds = uncontrolled_bounds)
        controlled_vars = tmp_contract.set_controlled_vars(controlled_vars, bounds = controlled_bounds)
        # print(uncontrolled_vars)
        # print(controlled_vars)

        # print("vehicle_num: {}".format(vehicle_num))
        # for region_param in region_params:
        #     print(region_param)

        # Find the guarantees formula and set the gurantees of the contract
        # Initialize guarantees
        guarantees_formula = "("
        # guarantees_formula = ""
        
        # Find goal guarantees
        guarantees_formula += "(F[0,{}] (({} == {}) & ({} == {})))".format(H, "{}_{}".format(data["dynamics"]["x"][0], vehicle_num), data["vehicle"]["target"][vehicle_num][0], "{}_{}".format(data["dynamics"]["x"][1], vehicle_num), data["vehicle"]["target"][vehicle_num][1])
        
        # Find no collision guarantees
        vehicle_width = data["vehicle"]["width"]
        # vehicle_length = data["vehicle"]["length"]
        if len(data["vehicle"]["id"]) >= 2:
            for i in data["vehicle"]["id"]:
                if i != vehicle_num:
                    guarantees_formula += " & (G[0,{}] (({} - {} >= {}) | ({} - {} >= {}) | ({} - {} >= {}) | ({} - {} >= {})))".format(H, 
                                            "{}_{}".format(data["dynamics"]["x"][0], vehicle_num), "{}_{}".format(data["dynamics"]["x"][0], i), vehicle_width, 
                                            "{}_{}".format(data["dynamics"]["x"][0], i), "{}_{}".format(data["dynamics"]["x"][0], vehicle_num), vehicle_width, 
                                            "{}_{}".format(data["dynamics"]["x"][1], vehicle_num), "{}_{}".format(data["dynamics"]["x"][1], i), vehicle_width, 
                                            "{}_{}".format(data["dynamics"]["x"][1], i), "{}_{}".format(data["dynamics"]["x"][1], vehicle_num), vehicle_width)
            guarantees_formula += ')'
        else:
            guarantees_formula = guarantees_formula[1:-1]

        # Set the contracts
        tmp_contract.set_guaran(guarantees_formula)
        
        # Saturate contract
        tmp_contract.checkSat()
        # print(tmp_contract)
        # print(guarantees_formula)
        
        # Add the contract
        tmp_solver.add_contract(tmp_contract)

        # Add the contract specifications
        tmp_solver.add_constraint(tmp_contract.guarantee)

        # Set objectives
        # TODO: Goal objectives
        # print(tmp_contract.guarantee)
        objective_func = gp.abs_(uncontrolled_vars[0]-uncontrolled_vars[1])

        # TODO: Region objectives
        region_params = np.array(data["vehicle"]["region"]["equation"][vehicle_num])

        # TODO: Fuel objectives
        
        # TODO: Set initial states. Make it automated not manual later.
        for var in uncontrolled_vars:
            if 'v_x' in var.name:
                tmp_solver.add_constraint(var == 10)
            elif 'v_y' in var.name:
                tmp_solver.add_constraint(var == 0)
            elif var.name == 'd_x_0':
                tmp_solver.add_constraint(var == 0)
            elif var.name == 'd_y_0':
                tmp_solver.add_constraint(var == 8)
            elif var.name == 'd_x_1':
                tmp_solver.add_constraint(var == 50)
            elif var.name == 'd_y_1':
                tmp_solver.add_constraint(var == 8)
            elif var.name == 'd_x_2':
                tmp_solver.add_constraint(var == 0)
            elif var.name == 'd_y_2':
                tmp_solver.add_constraint(var == 12)
        
        # Set Dynamics
        checked_ego = False
        for tmp_vehicle_num in data["vehicle"]["id"]: 
            # Build a linear system dynamics
            tmp_x = Vector(uncontrolled_vars[6*tmp_vehicle_num-2*checked_ego:6*tmp_vehicle_num+4-2*checked_ego])
            if vehicle_num == tmp_vehicle_num:
                tmp_u = Vector(controlled_vars)
                checked_ego = True
            else:
                tmp_u = Vector(uncontrolled_vars[6*tmp_vehicle_num+4-2*checked_ego: 6*tmp_vehicle_num+6-2*checked_ego])

            tmp_solver.add_dynamic(Next(tmp_x) == A * tmp_x + B * tmp_u)
        
        # Solve the problem using MILP solver
        solved = tmp_solver.solve()
        if solved:
            tmp_solver.print_solution()

env_load(10)
