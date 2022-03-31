# from copy import deepcopy
# import cplex
from posixpath import split
from gurobipy import GRB
import gurobipy as gp
import numpy as np
from scipy.stats import norm
from z3 import *

from pystl.contracts import *
from pystl.parser import *

M = 10**4
EPS = 10**-4
parser = Parser()

def split_var_time(str):
    idx = str.find('__')
    if idx == -1:
        return [str, -1]
    else:
        return [str[0:idx], int(str[idx+2:len(str)])]

def vec2expr(vector):
    """
    Translate all the entries in the matrix to expression object.

    :param matrix: [description]
    :type matrix: [type]
    :return: [description]
    :rtype: [type]
    """
    for i in range(len(vector)):
        vector[i] = entry2expr(vector[i])
    return vector

def mat2expr(matrix):
    """
    Translate all the entries in the matrix to expression object.

    :param matrix: [description]
    :type matrix: [type]
    :return: [description]
    :rtype: [type]
    """
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = entry2expr(matrix[i][j])
    return matrix

def entry2expr(entry):
    """Pystl expression to gurobi expression."""
    if isinstance(entry, str):
        out = parser(entry, 'expression')
    else:
        out = parser('{}'.format(entry), 'expression')
    return out

def expr2gurobiexpr(solver, expr):
    """Pystl expression to gurobi expression."""
    gurobi_expr = 0
    for i in range(len(expr.multipliers)):
        gurobi_term = expr.multipliers[i]
        for j in range(len(expr.var_list_list[i])):
            if expr.var_list_list[i][j] != 1:
                for _ in range(expr.power_list_list[i][j]):
                    try:
                        gurobi_term *= solver.model.getVarByName(expr.var_list_list[i][j])
                    except:
                        idx = solver.nondeter_vars.index(expr.var_list_list[i][j])
                        gurobi_term *= expr2gurobiexpr(solver.nondeter_vars_mean[idx])
        gurobi_expr += gurobi_term
    return gurobi_expr

class MILPSolver:
    """
    Attributes
        MILP_convex_solver
        b_vars
        b_var_num
    """
    __slots__ = ('model', 'contract', 'dynamics', 'horizon', 'node_vars', 'deter_vars', 'deter_vars_expr', 'nondeter_vars', 'nondeter_vars_expr', 'mode', 'soft_constraint_vars', 'verbose')

    def __init__(self, mode = "Boolean", verbose=False):
        assert(mode in ("Boolean", "Quantitative"))
        # print("====================================================================================")
        # print("Initializing MILP solver...")

        self.mode = mode
        self.verbose = verbose
        self.reset()

    def reset(self):
        # Initialize attributes
        self.contract = None
        self.dynamics = None
        self.horizon = [0, 0]
        self.node_vars = {}
        self.deter_vars = []
        self.deter_vars_expr = []
        self.nondeter_vars = []
        self.nondeter_vars_expr = []
        self.soft_constraint_vars = []

        # Initialize convex solver
        self.model = gp.Model() # Convex solver for solving the MILP convex problem
        self.model.setParam("OutputFlag", 0)
        self.model.setParam("NonConvex", 2)

    def add_contract(self, contract):
        """
        Adds a contract to the MILP solver.
        """
        # Add contract
        self.contract = contract

        # Find horizon
        [_, horizon_end_a] = contract.assumption.find_horizon()
        [_, horizon_end_g] = contract.guarantee.find_horizon()
        self.horizon = max(horizon_end_a, horizon_end_g)

        # Add deterministic variables
        for variable in contract.deter_var_list:
            for k in range(self.horizon+1):
                self.deter_vars.append("{}__{}".format(variable, k))
        self.deter_vars_expr = [0]*len(self.deter_vars)
            
        # Add deterministic variables
        for i, variable in enumerate(self.deter_vars):
            [var_str, _] = split_var_time(variable)
            idx = self.contract.deter_var_list.index(var_str)
            self.deter_vars_expr[i] = entry2expr(variable)
            self.model.addVar(name=variable, lb=self.contract.deter_var_bounds[idx][0], ub=self.contract.deter_var_bounds[idx][1], vtype=GRB.CONTINUOUS)
            self.model.update()

        # Add non-deterministic variables
        self.nondeter_vars_expr = [0]*len(contract.nondeter_var_list)*(self.horizon+1)
        
        for i, variable in enumerate(contract.nondeter_var_list):
            for k in range(self.horizon+1):
                self.nondeter_vars.append("{}__{}".format(variable, k))
                self.nondeter_vars_expr[i*(self.horizon+1)+k] = entry2expr("{}__{}".format(variable, k))

        # Add parametric variables
        for i, variable in enumerate(contract.param_var_list):
            self.model.addVar(lb=contract.param_var_bounds[i][0], ub=contract.param_var_bounds[i][1], vtype=GRB.CONTINUOUS, name=variable)
        self.model.update()

        # print(self.deter_vars)
        # print(self.deter_vars_expr)
        # print(self.nondeter_vars)
        # print(self.nondeter_vars_expr)
        # input()

    def add_dynamics(self, x = [], u = [], w = [], A = None, B = None, C = None, D = None, E = None):
        """
        Adds constraints for dynamics. 
        x_{k+1} = A*x_{k} + B*u_{k} + C*w_{k}

        :param x: [description], defaults to None
        :type x: [type], optional
        :param u: [description], defaults to None
        :type u: [type], optional
        :param w: [description], defaults to None
        :type w: [type], optional
        :param y: [description], defaults to None
        :type y: [type], optional
        :param A: [description], defaults to None
        :type A: [type], optional
        :param B: [description], defaults to None
        :type B: [type], optional
        :param C: [description], defaults to None
        :type C: [type], optional
        """
        # Define a function which recognizes a list of lists with all zero entries
        def allZero(listoflists):
            allZero = True
            for row in listoflists:
                if not all(elem == 0 for elem in row):
                    allZero = False
            return allZero

        # TODO: may have to modify when random variables are correlated
        # Check the dimensions of the matrices
        if A is not None:
            assert(len(A) == len(x))
            assert(len(A[0]) == len(x))
            mat2expr(A)

        if B is not None:
            assert(len(B) == len(x))
            assert(len(B[0]) == len(u))
            mat2expr(B)

        if C is not None:
            validC = allZero(C)
            assert(len(C) == len(x))
            assert(len(C[0]) == len(w))
            mat2expr(C)

        if D is not None:
            assert(len(D) == len(u))
            assert(len(D[0]) == len(x))
            mat2expr(D)

        # Add dynamics as a dictionary
        self.dynamics = {'x': x, 'u': u, 'w': w, 'A': A, 'B': B, 'C': C, 'D': D}

        # If w-vector exists and C is nonzero, convert all variables within x-vector to nondeter variables
        orig_nondeter_num = len(self.nondeter_vars)
        if len(w) != 0 and not validC:
            # Remove vars in x-vector from deter_var
            for var in x:
                for k in range(1, self.horizon+1):
                    idx = self.deter_vars.index("{}__{}".format(var, k))
                    del self.deter_vars[idx]
                    del self.deter_vars_expr[idx]
                    self.model.remove(self.model.getVarByName("{}__{}".format(var, k)))
                    self.nondeter_vars.append("{}__{}".format(var, k))
            self.nondeter_vars_expr= self.nondeter_vars_expr + [0]*len(x)*self.horizon

            # Find expression for each nondeter variable
            # TODO: Cov has to be better dealt with if two nondeter vars are correlated. Assuming independence for now.
            for k in range(1, self.horizon+1):
                for i in range(len(x)):
                    tmp_x = entry2expr(0)
                    idx1 = self.nondeter_vars.index("{}__{}".format(x[i], k))
                    for j in range(len(x)):
                        if repr(A[i][j]) != '0.0':
                            if k == 1:
                                idx2 = self.deter_vars.index("{}__{}".format(x[j], k-1))
                                tmp_x += A[i][j]*self.deter_vars_expr[idx2]
                            else:
                                idx2 = self.nondeter_vars.index("{}__{}".format(x[j], k-1))
                                tmp_x += A[i][j]*self.nondeter_vars_expr[idx2]

                    for j in range(len(u)):
                        if repr(B[i][j]) != '0.0':
                            idx2 = self.deter_vars.index("{}__{}".format(u[j], k-1))
                            tmp_x += B[i][j]*self.deter_vars_expr[idx2]

                    for j in range(len(w)):
                        if repr(C[i][j]) != '0.0':
                            if "{}__{}".format(w[j], k-1) in self.deter_vars:
                                idx2 = self.deter_vars.index("{}__{}".format(w[j], k-1))
                                tmp_x += C[i][j]*self.deter_vars_expr[idx2]
                            elif "{}__{}".format(w[j], k-1) in self.nondeter_vars:
                                idx2 = self.nondeter_vars.index("{}__{}".format(w[j], k-1))
                                tmp_x += C[i][j]*self.nondeter_vars_expr[idx2]

                    self.nondeter_vars_expr[idx1] = tmp_x
    
        else:
            for k in range(1, self.horizon+1):
                for i in range(len(x)):
                    tmp_x = entry2expr(0)
                    idx1 = self.deter_vars.index("{}__{}".format(x[i], k))
                    for j in range(len(x)):
                        if repr(A[i][j]) != '0.0':
                            tmp_x += A[i][j]*entry2expr("{}__{}".format(x[j], k-1))

                    for j in range(len(u)):
                        if repr(B[i][j]) != '0.0':
                            tmp_x += B[i][j]*entry2expr("{}__{}".format(u[j], k-1))
                    
                    self.deter_vars_expr[idx1] = tmp_x
                    
                    self.model.addConstr(expr2gurobiexpr(self, entry2expr("{}__{}".format(x[i], k))) == expr2gurobiexpr(self, tmp_x))
                    self.model.update

        # for k in range(1, self.horizon):
        #     for i in range(len(u)):
        #         tmp_x = entry2expr(0)
        #         idx1 = self.nondeter_vars.index("{}__{}".format(x[i], k))
        #         for j in range(len(x)):
        #             if repr(D[i][j]) != '0.0':
        #                 if k == 1:
        #                     idx2 = self.deter_vars.index("{}__{}".format(x[j], k-1))
        #                     tmp_x += D[i][j]*self.deter_vars_expr[idx2]
        #                 else:
        #                     idx2 = self.nondeter_vars.index("{}__{}".format(x[j], k-1))
        #                     tmp_x += D[i][j]*self.nondeter_vars_expr[idx2]

        #         self.nondeter_vars_expr[idx1] = tmp_x

        # print(self.deter_vars)
        # print(self.deter_vars_expr)
        # print(self.nondeter_vars)
        # print(self.nondeter_vars_expr)
        # input()

    def add_init_condition(self, formula):
        """
        Adds constraints for an initial condition.

        :param formula: [description]
        :type formula: [type]
        """
        # Parse the nontemporal formula
        if isinstance(formula, str):
            formula = parser(formula)[0][0]
        
        # Add constraints
        self.add_constraint(formula, name='b_{}'.format(len(self.node_vars)))

    def set_objective(self, sense='minimize'):
        # input()
        if sense == 'minimize':
            # try:
            #     self.model.remove(self.model.getConstrByName('UNSAT'))
            # except:
            #     pass
            self.model.setObjective(self.soft_constraint_vars[0], GRB.MINIMIZE)
            # if self.mode == 'Quantitative':
            #     self.model.addConstr(self.soft_constraint_vars[0] <= -EPS, 'SAT')
            # else:
            #     self.model.addConstr(self.soft_constraint_vars[0] == 0, 'SAT')
        else:
            # try:
            #     self.model.remove(self.model.getConstrByName('SAT'))
            # except:
            #     pass
            self.model.setObjective(self.soft_constraint_vars[0], GRB.MAXIMIZE)
            # if self.mode == 'Quantitative':
            #     # self.model.addConstr(self.soft_constraint_vars[0] >= 0, 'UNSAT')
            # else:
            #     self.model.addConstr(self.soft_constraint_vars[0] == 1, 'UNSAT')
        self.model.update()

    def solve(self):
        """ Solves the MILP problem """
        # Solve the optimization problem
        self.model.write('MILP.lp')
        self.model.optimize()
        # input()

        if self.status():
            if self.verbose:
                print("MILP solved.")
                self.print_solution()
            return True
        else:
            # self.model.computeIIS()
            # self.model.write("model.ilp")
            if self.verbose:
                print('There exists no solution.')
            return False

    def add_constraint(self, node, hard=True, name='b'):
        """
        Adds contraints of a StSTL formula to the solvers.
        """
        # Build the parse tree
        # node.printInfo()
        # print(node)
        copy_node = deepcopy(node)
        processed_node = copy_node.push_negation()
        # processed_node.printInfo()

        # Add the constraints
        if self.mode == 'Boolean':
            self.node_vars[name] = self.model.addVar(name=name, vtype=GRB.BINARY)
            if hard:
                self.model.addConstr(self.node_vars[name] == 1)
            else:
                self.soft_constraint_vars.append(self.node_vars[name])
        elif self.mode == 'Quantitative':
            self.node_vars[name] = self.model.addVar(name=name, lb=-M, ub=M, vtype=GRB.CONTINUOUS)
            if hard:
                self.model.addConstr(self.node_vars[name] >= 0)
            else:
                self.soft_constraint_vars.append(self.node_vars[name])
        self.model.update()

        # Add node constraints
        self.add_node_constraint(processed_node, top_node=True, name=name)

    def add_node_constraint(self, node, start_time = 0, end_time = 0, top_node=False, name='b'):
        """
        Adds convex constraints from a contract to a convex solver.
        """
        if top_node:
            b_name = name
        else:
            b_name = "{}_{}".format(name[0:3], len(self.node_vars))
            # 1. Add Boolean variables to the MILP convex solver
            if self.mode == 'Boolean':
                self.node_vars[b_name] = self.model.addVars(1, end_time+1, vtype=GRB.BINARY, name=b_name)
            elif self.mode == 'Quantitative':
                self.node_vars[b_name] = self.model.addVars(1, end_time+1, lb=-M, ub=M, vtype=GRB.CONTINUOUS, name=b_name)
            
            self.model.update()

        #  2. construct constraints according to the operation type
        if type(node) in (AP, stAP):
            self.add_ap_stap_constraint(node, self.node_vars[b_name], start_time, end_time, name=b_name)
        elif type(node) == temporal_unary: # Temporal unary operators
            self.add_temporal_unary_constraint(node, self.node_vars[b_name], start_time, end_time, name=b_name)
        elif type(node) == temporal_binary: # Temporal binary operator
            self.add_temporal_binary_constraint(node, self.node_vars[b_name], start_time, end_time, name=b_name)
        elif type(node) == nontemporal_unary: # Non-temporal unary operator
            self.add_nontemporal_unary_constraint(node, self.node_vars[b_name], start_time, end_time, name=b_name)
        elif type(node) == nontemporal_multinary: # Non-temporal multinary operator
            self.add_nontemporal_multinary_constraint(node, self.node_vars[b_name], start_time, end_time, name=b_name)
        elif type(node) == boolean:
            self.add_boolean_constraint(node, self.node_vars[b_name], start_time, end_time)
        
        return self.node_vars[b_name]

    def add_ap_stap_constraint(self, node, parent_vars, start_time = 0, end_time = 0, name='b'):
        # print(node)
        if type(node) == AP: # Add constraints for an atomic predicate
            if type(parent_vars) == gp.Var:
                constr = entry2expr(0)
                for i in range(len(node.multipliers)):
                    term = entry2expr(node.multipliers[i])
                    for j, var in enumerate(node.var_list_list[i]):
                        if var != 1:
                            if var in self.contract.param_var_list:
                                term *= entry2expr(var)
                            else:
                                for _ in range(int(node.power_list_list[i][j])):
                                    idx = self.deter_vars.index("{}__0".format(var))
                                    term *= self.deter_vars_expr[idx]
                    constr += term

                # Pystl expression to gurobi expression
                gurobi_expr = expr2gurobiexpr(self, constr)

                # Add constraints
                if self.mode == 'Boolean':
                    self.model.addConstr(M * (1 - parent_vars) >= gurobi_expr)
                    self.model.addConstr(EPS - M * parent_vars <= gurobi_expr)
                    if node.equal:
                        self.model.addConstr(M * (1 - parent_vars) >= -gurobi_expr)
                        self.model.addConstr(EPS - M * parent_vars <= -gurobi_expr)
                else:
                    self.model.addConstr(parent_vars == -gurobi_expr)
                    if node.equal:
                        self.model.addConstr(gurobi_expr == 0)
            
            else:
                # print(self.deter_vars)
                for t in range(start_time, end_time+1):
                    constr = entry2expr(0)
                    for i in range(len(node.multipliers)):
                        term = entry2expr(node.multipliers[i])
                        for j, var in enumerate(node.var_list_list[i]):
                            if var != 1:
                                if var in self.contract.param_var_list:
                                    term *= entry2expr(var)
                                else:
                                    for _ in range(int(node.power_list_list[i][j])):
                                        idx = self.deter_vars.index("{}__{}".format(var, t))
                                        term *= self.deter_vars_expr[idx]
                        constr += term

                    # Pystl expression to gurobi expression
                    gurobi_expr = expr2gurobiexpr(self, constr)

                    # Add constraints
                    if self.mode == 'Boolean':
                        self.model.addConstr(M * (1 - parent_vars[0,t]) >= gurobi_expr)
                        self.model.addConstr(EPS - M * parent_vars[0,t] <= gurobi_expr)
                        if node.equal:
                            self.model.addConstr(M * (1 - parent_vars[0,t]) >= -gurobi_expr)
                            self.model.addConstr(EPS - M * parent_vars[0,t] <= -gurobi_expr)
                    else:
                        self.model.addConstr(parent_vars[0,t] == -gurobi_expr)
                        if node.equal:
                            self.model.addConstr(gurobi_expr == 0)
        
        elif type(node) == stAP: # Add constraints for a stochastic atomic predicates
            # Function for finding mean and variance of an expression
            # print(node)
            def expr2meanvar(expr):
                mean = entry2expr(0)
                variance = entry2expr(0)
                for i in range(len(expr.multipliers)):
                    term_mean = entry2expr(expr.multipliers[i])
                    term_variance = entry2expr(expr.multipliers[i]*expr.multipliers[i])
                    for j, var in enumerate(expr.var_list_list[i]):
                        if var != 1:
                            if var in self.contract.param_var_list:
                                term_mean *= entry2expr(var)
                                term_variance = entry2expr(0)
                            else:
                                for _ in range(int(expr.power_list_list[i][j])):
                                    if var in self.nondeter_vars:
                                        [varname, _] = split_var_time(var)
                                        idx = self.contract.nondeter_var_list.index(varname)
                                        term_mean *= entry2expr(self.contract.nondeter_var_mean[idx])
                                        term_variance *= entry2expr(self.contract.nondeter_var_cov[idx][idx])
                                    else:
                                        term_mean *= entry2expr(var)
                                        term_variance = entry2expr(0)
                        else:
                            term_variance = entry2expr(0)
                    mean += term_mean
                    variance += term_variance
                return [mean, variance]

            # Fetch the probability expression for the node
            prob = entry2expr(0)
            for i in range(len(node.prob_multipliers)):
                term = entry2expr(node.prob_multipliers[i])
                for j, var in enumerate(node.prob_var_list_list[i]):
                    if var != 1:
                        for _ in range(int(node.power_list_list[i][j])):
                            term *= entry2expr(var)
                    prob += term

            if len(prob.variables) == 0:
                # Handle extreme cases
                if prob.multipliers[0] == 0:
                    prob = entry2expr(0.0000000000000001)
                elif prob.multipliers[0] == 1:
                    prob = entry2expr(0.9999999999999999)
            else:
                # Find lower and upper bounds for the probability threshold
                p_low_mat = []
                p_high_mat = []
                p_multipliers = []
                for i, param_var in enumerate(self.contract.param_var_list):
                    p_low_mat.append(self.contract.param_var_bounds[i][0])
                    p_high_mat.append(self.contract.param_var_bounds[i][1])
                    if [param_var] in node.prob_var_list_list:
                        p_multipliers.append(node.prob_multipliers[node.prob_var_list_list.index([param_var])])
                    else:
                        p_multipliers.append(0)
                
                p_low = 0
                p_high = 0
                for i in range(len(p_low_mat)):
                    if p_multipliers[i] < 0:
                        p_low += p_high_mat[i]*p_multipliers[i]
                        p_high += p_low_mat[i]*p_multipliers[i]
                    else:
                        p_low += p_low_mat[i]*p_multipliers[i]
                        p_high += p_high_mat[i]*p_multipliers[i]

                if [1] in node.prob_var_list_list:
                    idx = node.prob_var_list_list.index([1])
                    p_low += node.prob_multipliers[idx]
                    p_high += node.prob_multipliers[idx]

                # Handle extreme cases
                if p_low <= 0:
                    p_low = 10**-16
                if p_high >= 1:
                    p_high = 1- 10**-16
            
                # print((p_low, p_high))

            if type(parent_vars) == gp.Var:
                constr = entry2expr(0)
                expr = entry2expr(0)
                for i in range(len(node.multipliers)):
                    term_expr = entry2expr(node.multipliers[i])
                    for j, var in enumerate(node.var_list_list[i]):
                        if var != 1:
                            if var in self.contract.param_var_list:
                                term_expr *= entry2expr(var)
                            else:
                                for _ in range(int(node.power_list_list[i][j])):
                                    if "{}__0".format(var) in self.nondeter_vars:
                                        idx = self.nondeter_vars.index("{}__0".format(var))
                                        term_expr *= self.nondeter_vars_expr[idx]
                                    else:
                                        term_expr *= entry2expr("{}__0".format(var))
                    expr += term_expr

                [mean, variance] = expr2meanvar(expr)

                # print("Probability: {}".format(prob))
                # print("Expression: {}".format(expr))
                # print("Mean: {}".format(mean))
                # print("Variance: {}".format(variance))
                # input()
                
                # Find chance constraint
                probIsNum = False
                if len(prob.variables) == 0:
                    probIsNum = True
                varIsNum = False
                if len(variance.variables) == 0:
                    varIsNum = True

                if probIsNum and varIsNum:
                    # Find LHS of the constraint 
                    constr = mean + entry2expr(norm.ppf(float(repr(prob))))*entry2expr(math.sqrt(float(repr(variance))))
                
                elif not probIsNum and varIsNum:
                    # Find LHS of the constraint 
                    if p_low >= 0.5:
                        constr = mean + (entry2expr(norm.ppf(p_low)) + entry2expr((norm.ppf(p_high)-norm.ppf(p_low))/(p_high-p_low))*(prob-entry2expr(p_low)))*entry2expr(math.sqrt(float(repr(variance))))
                    else:
                        constr = mean + (entry2expr(norm.ppf((p_low+p_high)/2)) + entry2expr(1/norm.pdf(norm.ppf((p_low+p_high)/2)))*(prob-entry2expr((p_low+p_high)/2)))*entry2expr(math.sqrt(float(repr(variance))))
                
                elif probIsNum and not varIsNum:
                    # Add anxiliary variables and constraints
                    tmp_variance = self.model.addVar(name='{}_variance__0'.format(name), ub = 10**4, vtype=GRB.CONTINUOUS)
                    tmp_sigma = self.model.addVar(name='{}_sigma__0'.format(name), ub = 10**4, vtype=GRB.CONTINUOUS)
                    self.model.addConstr(tmp_variance == expr2gurobiexpr(self, variance))
                    self.model.addGenConstrPow(tmp_variance, tmp_sigma, 0.5)
                    self.model.update()
                    
                    # Find LHS of the constraint 
                    constr = mean + entry2expr(norm.ppf(float(repr(prob))))*parser('{}_sigma__0'.format(name), 'expression')
                
                elif not probIsNum and not varIsNum:
                    # Add anxiliary variables and constraints
                    tmp_variance = self.model.addVar(name='{}_variance__0'.format(name), ub = 10**4, vtype=GRB.CONTINUOUS)
                    tmp_sigma = self.model.addVar(name='{}_sigma__0'.format(name), ub = 10**4, vtype=GRB.CONTINUOUS)
                    self.model.addConstr(tmp_variance == expr2gurobiexpr(self, variance))
                    self.model.addGenConstrPow(tmp_variance, tmp_sigma, 0.5)
                    self.model.update()
                    
                    # Find LHS of the constraint 
                    if p_low >= 0.5:
                        constr = mean + (entry2expr(norm.ppf(p_low)) + entry2expr((norm.ppf(p_high)-norm.ppf(p_low))/(p_high-p_low))*(prob-entry2expr(p_low)))*parser('{}_sigma__0'.format(name), 'expression')
                    else:
                        constr = mean + (entry2expr(norm.ppf((p_low+p_high)/2)) + entry2expr(1/norm.pdf(norm.ppf((p_low+p_high)/2)))*(prob-entry2expr((p_low+p_high)/2)))*parser('{}_sigma__0'.format(name), 'expression')

                # Pystl expression to gurobi expression
                gurobi_expr = expr2gurobiexpr(self, constr)

                # Add constraints
                if self.mode == 'Boolean':
                    self.model.addConstr(M * (1 - parent_vars) >= gurobi_expr)
                    self.model.addConstr(EPS - M * parent_vars <= gurobi_expr)
                else:
                    self.model.addConstr(parent_vars == -gurobi_expr)
                self.model.update()
                
            else:
                for t in range(start_time, end_time+1):
                    constr = entry2expr(0)
                    expr = entry2expr(0)
                    for i in range(len(node.multipliers)):
                        term_expr = entry2expr(node.multipliers[i])
                        for j, var in enumerate(node.var_list_list[i]):
                            if var != 1:
                                if var in self.contract.param_var_list:
                                    term_expr *= entry2expr(var)
                                else:
                                    for _ in range(int(node.power_list_list[i][j])):
                                        if "{}__{}".format(var, t) in self.nondeter_vars:
                                            idx = self.nondeter_vars.index("{}__{}".format(var, t))
                                            term_expr *= self.nondeter_vars_expr[idx]
                                        else:
                                            term_expr *= entry2expr("{}__{}".format(var, t))
                        expr += term_expr

                    [mean, variance] = expr2meanvar(expr)

                    # print("At time: {}".format(t))
                    # print("Probability: {}".format(prob))
                    # print("Expression: {}".format(expr))
                    # print("Mean: {}".format(mean))
                    # print("Variance: {}".format(variance))
                    # input()
                
                    # Find chance constraint
                    probIsNum = False
                    if len(prob.variables) == 0:
                        probIsNum = True
                    varIsNum = False
                    if len(variance.variables) == 0:
                        varIsNum = True

                    if probIsNum and varIsNum:
                        # Find LHS of the constraint 
                        constr = mean + entry2expr(norm.ppf(float(repr(prob))))*entry2expr(math.sqrt(float(repr(variance))))
                    
                    elif not probIsNum and varIsNum:
                        # Find LHS of the constraint 
                        if p_low >= 0.5:
                            constr = mean + (entry2expr(norm.ppf(p_low)) + entry2expr((norm.ppf(p_high)-norm.ppf(p_low))/(p_high-p_low))*(prob-entry2expr(p_low)))*entry2expr(math.sqrt(float(repr(variance))))
                        else:
                            constr = mean + (entry2expr(norm.ppf((p_low+p_high)/2)) + entry2expr(1/norm.pdf(norm.ppf((p_low+p_high)/2)))*(prob-entry2expr((p_low+p_high)/2)))*entry2expr(math.sqrt(float(repr(variance))))
                    
                    elif probIsNum and not varIsNum:
                        # Add anxiliary variables and constraints
                        tmp_variance = self.model.addVar(name='{}_variance__{}'.format(name, t), ub = 10**4, vtype=GRB.CONTINUOUS)
                        tmp_sigma = self.model.addVar(name='{}_sigma__{}'.format(name, t), ub = 10**4, vtype=GRB.CONTINUOUS)
                        self.model.addConstr(tmp_variance == expr2gurobiexpr(self, variance))
                        self.model.addGenConstrPow(tmp_variance, tmp_sigma, 0.5)
                        self.model.update()
                        
                        # Find LHS of the constraint 
                        constr = mean + entry2expr(norm.ppf(float(repr(prob))))*entry2expr('{}_sigma__{}'.format(name, t))
                    
                    elif not probIsNum and not varIsNum:
                        # Add anxiliary variables and constraints
                        tmp_variance = self.model.addVar(name='{}_variance__{}'.format(name, t), ub = 10**4, vtype=GRB.CONTINUOUS)
                        tmp_sigma = self.model.addVar(name='{}_sigma__{}'.format(name, t), ub = 10**4, vtype=GRB.CONTINUOUS)
                        self.model.addConstr(tmp_variance == expr2gurobiexpr(self, variance))
                        self.model.addGenConstrPow(tmp_variance, tmp_sigma, 0.5)
                        self.model.update()
                        
                        # Find LHS of the constraint 
                        if p_low >= 0.5:
                            constr = mean + (entry2expr(norm.ppf(p_low)) + entry2expr((norm.ppf(p_high)-norm.ppf(p_low))/(p_high-p_low))*(prob-entry2expr(p_low)))*entry2expr('{}_sigma__{}'.format(name, t))
                        else:
                            constr = mean + (entry2expr(norm.ppf((p_low+p_high)/2)) + entry2expr(1/norm.pdf(norm.ppf((p_low+p_high)/2)))*(prob-entry2expr((p_low+p_high)/2)))*entry2expr('{}_sigma__{}'.format(name, t))

                    # Pystl expression to gurobi expression
                    gurobi_expr = expr2gurobiexpr(self, constr)

                    # Add constraints
                    if self.mode == 'Boolean':
                        self.model.addConstr(M * (1 - parent_vars[0,t]) >= gurobi_expr)
                        self.model.addConstr(EPS - M * parent_vars[0,t] <= gurobi_expr)
                    else:
                        self.model.addConstr(parent_vars[0,t] == -gurobi_expr)
                    self.model.update()

    def add_temporal_unary_constraint(self, node, parent_vars, start_time = 0, end_time = 0, name='b'):
        # Fetch node variables from the child node
        child_vars = self.add_node_constraint(node.children_list[0], start_time=start_time+node.interval[0], end_time=end_time+node.interval[1], name=name)

		# Encode the logic
        if node.operator == 'G': # Globally
            if type(parent_vars) == gp.Var:
                if self.mode == 'Boolean':
                    self.model.addConstr(parent_vars == gp.and_(child_vars.values()[node.interval[0]:node.interval[1]+1]))
                else:
                    self.model.addConstr(parent_vars == gp.min_(child_vars.values()[node.interval[0]:node.interval[1]+1]))
            else:
                for i in range(start_time, end_time+1):
                    if self.mode == 'Boolean':
                        self.model.addConstr(parent_vars[0,i] == gp.and_(child_vars.values()[i+node.interval[0]:i+node.interval[1]+1]))
                    else:
                        self.model.addConstr(parent_vars[0,i] == gp.min_(child_vars.values()[i+node.interval[0]:i+node.interval[1]+1]))
        
        elif node.operator == 'F': # Finally
            if type(parent_vars) == gp.Var:
                if self.mode == 'Boolean':
                    self.model.addConstr(parent_vars == gp.or_(child_vars.values()[node.interval[0]:node.interval[1]+1]))
                else:
                    self.model.addConstr(parent_vars == gp.max_(child_vars.values()[node.interval[0]:node.interval[1]+1]))
            else:
                for i in range(start_time, end_time+1):
                    if self.mode == 'Boolean':
                        self.model.addConstr(parent_vars[0,i] == gp.or_(child_vars.values()[i+node.interval[0]:i+node.interval[1]+1]))
                    else:
                        self.model.addConstr(parent_vars[0,i] == gp.max_(child_vars.values()[i+node.interval[0]:i+node.interval[1]+1]))
        
        self.model.update()

    # TODO: Not implemented
    def add_temporal_binary_constraint(self, node, parent_vars, start_time = 0, end_time = 0, name='b'):
        # Fetch node variables from the child node
        child_vars1 = self.add_node_constraint(node.children_list[0], start_time=start_time, end_time=end_time+node.interval[1]-1, name=name)
        child_vars2 = self.add_node_constraint(node.children_list[1], start_time=start_time+node.interval[0], end_time=end_time+node.interval[1], name=name)

        # Encode the logic
        # for t in range(start_time, end_time):
        #     aux_variables = []
        #     for t_i in range(t + node.interval[0], t + node.interval[1] + 1):
        #         if self.mode == 'Boolean':
        #             aux_variable = self.model_add_binary_variable_by_name("node_{}_aux_{}_{}".format(node.idx, t, t_i))
        #         elif self.mode == 'Quantitative':
        #             aux_variable = self.model_add_continuous_variable_by_name("node_{}_aux_{}_{}".format(node.idx, t, t_i))
        #         else: assert(False)

        #         if node.ast_type == 'U': # Globally
        #             self.model_add_constraint_and(aux_variable, self.node_variable[node.formula_list[0].idx, t:t_i].tolist() + [self.node_variable[node.formula_list[1].idx, t_i]])
        #         elif node.ast_type == 'R': # Globally
        #             self.model_add_constraint_or(aux_variable, self.node_variable[node.formula_list[0].idx, t:t_i].tolist() + [self.node_variable[node.formula_list[1].idx, t_i]])
        #         else: assert(False)

        #         aux_variables.append(aux_variable)

        #     if node.ast_type == 'U': # Globally
        #         # Add anxiliary Boolean constraints to MILP convex solver
        #         self.model_add_constraint_or(self.node_variable[node.idx, t], aux_variables)
        #     elif node.ast_type == 'R': # Globally
        #         self.model_add_constraint_and(self.node_variable[node.idx, t], aux_variables)

    def add_nontemporal_unary_constraint(self, node, parent_vars, start_time = 0, end_time = 0, name='b'):
        # Fetch node variables from the child node
        child_vars = self.add_node_constraint(node.children_list[0], start_time=start_time, end_time=end_time+1, name=name)

        # Encode the logic 
        if type(parent_vars) == gp.Var:
            self.model.addConstr(parent_vars == child_vars[0, 1])
        else:
            for t in range(start_time, end_time+1):
                self.model.addConstr(parent_vars[0, t] == child_vars[0, t+1])
        self.model.update()

    def add_nontemporal_multinary_constraint(self, node, parent_vars, start_time = 0, end_time = 0, name='b'):
        # Fetch node variables from the child node
        child_vars_list = []
        for children in node.children_list:
            child_vars_list.append(self.add_node_constraint(children, start_time=start_time, end_time=end_time, name=name))

        # Encode the logic 
        if type(parent_vars) == gp.Var:
            tmp_list = []
            for child_vars in child_vars_list:
                tmp_list.append(child_vars[0,0])
            if node.operator == '&': # AND
                if self.mode == 'Boolean':
                    self.model.addConstr(parent_vars == gp.and_(tmp_list))
                else:
                    self.model.addConstr(parent_vars == gp.min_(tmp_list))
            elif node.operator == '|': # OR
                if self.mode == 'Boolean':
                    self.model.addConstr(parent_vars == gp.or_(tmp_list))
                else:
                    self.model.addConstr(parent_vars == gp.max_(tmp_list))
            self.model.update()
        else:
            for t in range(start_time, end_time+1):
                tmp_list = []
                for child_vars in child_vars_list:
                    tmp_list.append(child_vars[0,t])
                if node.operator == '&': # AND
                    if self.mode == 'Boolean':
                        self.model.addConstr(parent_vars[0,t] == gp.and_(tmp_list))
                    else:
                        self.model.addConstr(parent_vars[0,t] == gp.min_(tmp_list))
                elif node.operator == '|': # OR
                    if self.mode == 'Boolean':
                        self.model.addConstr(parent_vars[0,t] == gp.or_(tmp_list))
                    else:
                        self.model.addConstr(parent_vars[0,t] == gp.max_(tmp_list))
            self.model.update()

    def add_boolean_constraint(self, node, parent_vars, start_time = 0, end_time = 0):
        # Encode the logic
        if node.formula == 'True':
            if type(parent_vars) == gp.Var:
                if self.mode == 'Boolean':
                    self.model.addConstr(parent_vars == 1)
                else:
                    self.model.addConstr(parent_vars == M)
            else:
                for t in range(start_time, end_time+1):
                    if self.mode == 'Boolean':
                        self.model.addConstr(parent_vars[0,t] == 1)
                    else:
                        self.model.addConstr(parent_vars[0,t] == M)
        elif node.formula == 'False':
            if type(parent_vars) == gp.Var:
                if self.mode == 'Boolean':
                    self.model.addConstr(parent_vars == 0)
                else:
                    self.model.addConstr(parent_vars == -M)
            else:
                for t in range(start_time, end_time+1):
                    if self.mode == 'Boolean':
                        self.model.addConstr(parent_vars[0,t] == 0)
                    else:
                        self.model.addConstr(parent_vars[0,t] == -M)
        self.model.update()

    def status(self):
        return self.model.getAttr("Status") == 2

    def print_solution(self, node_var_print=False):
        # Print node variables
        if node_var_print:
            for var_name, gurobi_vars in self.node_vars.items():
                if type(gurobi_vars) == gp.Var:
                    print("{}: {}".format(var_name, gurobi_vars.x))
                else:
                    for var_idx, gurobi_var in gurobi_vars.items():
                        print("{}[{}]: {}".format(var_name, var_idx[1], gurobi_var.x))

        # Print contract variables
        for var_name in self.nondeter_vars:
            [var, time] = split_var_time(var_name)
            if time == 0 and var not in self.dynamics['w']:
                print("{}: {}".format(var_name, self.model.getVarByName(var_name).x))
    
        # Print contract variables
        for var_name in self.deter_vars:
            print("{}: {}".format(var_name, self.model.getVarByName(var_name).x))
        print()
    
    # TODO: Fix here
    def fetch_solution(self, controlled_vars, length=1):
        """ 
        Fetches the values of the controlled variables, if available.
        """
        # Initialize output
        output = []

        # Find the output
        #  print(controlled_vars)
        #  print(self.contract_variable[controlled_vars[0].idx, 0])
        #  input()
        for var in controlled_vars:
            output.append([v.x for v in self.contract_variable[var.idx, 0: length]])
                    
        return output