from copy import deepcopy
import numpy as np
import random
import matplotlib.pyplot as plt
from pystl.variable import DeterVar, NondeterVar, M, EPS
from pystl.parser import P, G, F, true, false, ASTObject, Parser
from pystl.core import SMCSolver, MILPSolver


class contract:
    """
    A contract class for defining a contract object.

    :param id: An id of the contract
    :type id: str
    """

    __slots__ = ('id', 'deter_var_list', 'deter_var_name2id', 'nondeter_var_list', 'nondeter_var_name2id', 'nondeter_var_mean', 'nondeter_var_cov', 'assumption', 'guarantee', 'sat_guarantee', 'isSat')

    def __init__(self, id = ''):
        """ Constructor method """
        self.id = id
        self.deter_var_list             = [None]
        self.deter_var_name2id          = {None: 0}
        self.nondeter_var_list          = []
        self.nondeter_var_name2id       = {}
        self.nondeter_var_mean          = np.empty(0)
        self.nondeter_var_cov           = np.empty(0)
        self.assumption                 = true
        self.guarantee                  = false
        self.sat_guarantee              = false
        self.isSat                      = False

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
        res = []
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
        res = []
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
        assert(mean.shape == (len(var_names),))
        assert(cov.shape == (len(var_names), len(var_names)))
        res = []
        for i, name in enumerate(var_names):
            data = NondeterVar(name, len(self.nondeter_var_list), data_type = dtypes[i] if dtypes != None else 'GAUSSIAN')
            self.nondeter_var_list.append(data)
            self.nondeter_var_name2id[name] = len(self.nondeter_var_name2id)

            res.append(data)
        self.nondeter_var_mean = mean
        self.nondeter_var_cov = cov
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
        res = []
        for i, name in enumerate(param_names):
            data = DeterVar(name, len(self.deter_var_list), "parameter", data_type = dtypes[i] if dtypes is not None else 'CONTINUOUS', bound = bounds[i] if bounds is not None else None)
            self.deter_var_list.append(data)
            self.deter_var_name2id[name] = len(self.deter_var_name2id)

            res.append(data)
        return res

    def set_assume(self, assumption):
        """
        Sets the assumption of the contract

        :param assumption: An STL or StSTL formula which characterizes the assumption set of the contract
        :type assumption: str
        """
        if (isinstance(assumption, str)):
            parser = Parser(self)
            self.assumption = parser(assumption)
        elif (isinstance(assumption, ASTObject)):
            self.assumption = assumption
        else: assert(False)

    def set_guaran(self, guarantee):
        """
        Sets the guarantee of the contract

        :param guarantee: An STL or StSTL formula which characterizes the guarantee set of the contract
        :type guarantee: str
        """
        if (isinstance(guarantee, str)):
            parser = Parser(self)
            self.guarantee = parser(guarantee)
        elif (isinstance(guarantee, ASTObject)):
            self.guarantee = guarantee
        else: assert(False)

    def saturate(self):
    #      """ Saturates the contract """
        if self.isSat:
            return
        else:
            self.isSat = True
            assumption = deepcopy(self.assumption)
            guarantee = deepcopy(self.guarantee)
            self.sat_guarantee = assumption.implies(guarantee)
    def checkCompat(self, print_sol=False):
        """ Checks compatibility of the contract """
        # Build a MILP Solver
        print("Checking compatibility of the contract {}...".format(self.id))
        solver = MILPSolver()
        # Add a contract
        self.saturate()
        solver.add_contract(self)
        solver.add_constraint(self.assumption)
        # Solve the problem
        solved = solver.solve()
        # Print the solution
        if solved:
            print("Contract {} is compatible.\n".format(self.id))
            if print_sol:
                solver.print_solution()
        else:
            print("Contract {} is not compatible.\n".format(self.id))
    def checkConsis(self, print_sol=False):
        """ Checks consistency of the contract """
#          # Build a MILP Solver
        print("Checking consistency of the contract {}...".format(self.id))
        solver = MILPSolver()
#
        # Add a contract
        self.saturate()
        solver.add_contract(self)
        solver.add_constraint(self.sat_guarantee)
        # Solve the problem
        solved = solver.solve()
        # Print the solution
        if solved:
            print("Contract {} is consistent.\n".format(self.id))
            if print_sol:
                solver.print_solution()
        else:
            print("Contract {} is not consistent.\n".format(self.id))
    def checkFeas(self, print_sol=False, verbose = True):
        """ Checks feasibility of the contract """
#          # Build a MILP Solver
        if verbose:
            print("Checking feasibility of the contract {}...".format(self.id))
        solver = MILPSolver()
#
        # Add a contract
        self.saturate()
        solver.add_contract(self)
        solver.add_constraint(self.assumption & self.guarantee)
        # Solve the problem
        solved = solver.solve()
        # Print the solution
        if solved and verbose:
            print("Contract {} is feasible.\n".format(self.id))
            if print_sol:
                print("Printing a behavior that satisfies both the assumption and guarantee of the contract {}...".format(self.id))
                solver.print_solution()
        elif not solved and verbose:
            print("Contract {} is not feasible.\n".format(self.id))
        return solved
    def checkRefine(self, contract2refine, print_sol=False):
        """ Checks whether contract2refine refines the contract """
        # Build a MILP Solver
        c1 = deepcopy(self)
        (deter_id_map, nondeter_id_map) = c1.merge_contract(contract2refine)
        #  print(c1)
        #  print(contract2refine)
        #  print(deter_id_map, nondeter_id_map)

        print("Checking whether contract {} refines contract {}...".format(self.id, contract2refine.id))
        solver = MILPSolver()
        solver.add_contract(self)

        assumption1 = self.assumption
        assumption2 = deepcopy(contract2refine.assumption)
        assumption2.transform(deter_id_map, nondeter_id_map)

        solver.add_constraint(~(assumption2.implies(assumption1)))
        print("Checking condition 1 for refinement...")
        solved = solver.solve()

        # Print the counterexample
        if solved:
            print("Condition 1 for refinement violated. Contract {} does not refine contract {}.\n".format(self.id, contract2refine.id))
            if print_sol:
                solver.print_solution()
            return

        # Resets a MILP Solver
        solver.reset()
        solver.add_contract(self)

        guarantee1 = self.guarantee
        guarantee2 = deepcopy(contract2refine.guarantee)
        guarantee2.transform(deter_id_map, nondeter_id_map)
        solver.add_constraint(~(guarantee1.implies(guarantee2)))

        # Solve the problem
        print("Checking condition 2 for refinement...")
        solved = solver.solve()

        # Print the counterexample
        if solved:
            print("Condition 2 for refinement violated. Contract {} does not refine contract {}.\n".format(self.id, contract2refine.id))
            if print_sol:
                print("Printing a counterexample which violates condition 2 for refinement...")
                solver.print_solution()
            return

        print("Contract {} refines {}.\n".format(self.id, contract2refine.id))
    def merge_contract(self, contract):
        #  determinate variable
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
    def find_opt_param(self, objective, N=100):
        """ Find an optimal set of parameters for a contract given an objective function. """
        # Build a MILP Solver
        print("Finding an optimal set of parameters for contract {}...".format(self.id))


        variable = [False]
        bounds = []
        for v in self.deter_var_list[1:]:
            if v.var_type == 'parameter':
                variable.append(True)
                bounds.append(v.bound)
            else:
                variable.append(False)
        variable = np.array(variable)
        bounds = np.array(bounds)
        # Sample the parameters N times
        sampled_param = np.random.rand(N, len(bounds))
        sampled_param *= (bounds[:,1] - bounds[:,0])
        sampled_param += bounds[:,0]
        #  print(variable)
        #  print(bounds)
        #  print(sampled_param)
        #  input()
        def change(data, variable, values):
            #  print(data)
            #  print(variable)
            #  print(values)
            parameter = data[variable[:len(data)]] 
            data[0] += np.sum(parameter * values[:len(parameter)])
            data[variable[:len(data)]] = 0
        def traverse(node, variable, values):
            if node.ast_type == 'AP':
                change(node.expr.deter_data, variable, values)
            elif node.ast_type == 'StAP':
                #  print(node.prob)
                change(node.expr.deter_data, variable, values)
                change(node.prob.deter_data, variable, values)
            else:
                for f in node.formula_list:
                    traverse(f, variable, values)

        # Build a deepcopy of the contract
        c = deepcopy(self)

        x_id = np.argmax(variable)
        y_id = x_id + 1 + np.argmax(variable[x_id + 1:])

        fig = plt.figure()
        plt.xlabel(self.deter_var_list[x_id].name)
        plt.ylabel(self.deter_var_list[y_id].name)
        plt.xlim(self.deter_var_list[x_id].bound[0], self.deter_var_list[x_id].bound[1])
        plt.ylim(self.deter_var_list[y_id].bound[0], self.deter_var_list[y_id].bound[1])

        for i in range(N):
            #  print(sampled_param[i])
            c.assumption = deepcopy(self.assumption)
            traverse(c.assumption, variable, sampled_param[i])
            c.guarantee = deepcopy(self.guarantee)
            traverse(c.guarantee, variable, sampled_param[i])
            #  print(c)
            #  input()
            if c.checkFeas(print_sol = False, verbose = False):
                plt.plot(sampled_param[i, 0], sampled_param[i, 1], 'go')
            else:
                plt.plot(sampled_param[i, 0], sampled_param[i, 1], 'ro')

        plt.savefig('test.jpg')
    def printInfo(self):
        print(str(self))
    def __str__(self):
        """ Prints information of the contract """
        res = ''
        res += "Contract ID: {}".format(self.id)
        for v in self.deter_var_list[1:]:
            res += '\n  '
            res += "\n    ".join(str(v).splitlines())
        for v in self.nondeter_var_list:
            res += '\n  '
            res += "\n    ".join(str(v).splitlines())
        res += "\n"
        res += "    mean: {}\n".format(self.nondeter_var_mean)
        res += "    cov: {}\n".format(self.nondeter_var_cov)
        res += "  Assumption: {}\n".format(self.assumption)
        res += "  Guarantee: {}\n".format(self.guarantee)
        res += "  Saturated Guarantee: {}\n".format(self.sat_guarantee)
        res += "  isSat: {}\n".format(self.isSat)
        return res
    def __repr__(self):
        """ Prints information of the contract """
        """ Prints information of the contract """
        res = ''
        res += "Contract ID: {}".format(self.id)
        for v in self.deter_var_list[1:]:
            res += '\n  '
            res += "\n    ".join(repr(v).splitlines())
        for v in self.nondeter_var_list:
            res += '\n  '
            res += "\n    ".join(repr(v).splitlines())
        res += "\n"
        res += "  Assumption: {}\n".format(self.assumption)
        res += "  Guarantee: {}\n".format(self.guarantee)
        res += "  Saturated Guarantee: {}\n".format(self.sat_guarantee)
        res += "  isSat: {}\n".format(self.isSat)
        return res
#
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
    c1.saturate()
    c2.saturate()

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
#
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
    c1.saturate()
    c2.saturate()

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
    c.saturate()
    c2.saturate()

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

    c.saturate()
    c2.saturate()

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
