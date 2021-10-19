from copy import deepcopy
import cplex
from gurobipy import GRB
import gurobipy as gp
import numpy as np
from scipy.stats import norm
from z3 import *

from pystl.contracts import *
from pystl.parser import *

M = 10**4
EPS = 10**-4

class MILPSolver:
    """

    Attributes
        MILP_convex_solver
        b_vars
        b_var_num
    """
    __slots__ = ('model', 'interval', 'node_vars', 'contract_vars', 'mode', 'objective', 'verbose')

    def __init__(self, mode = "Boolean", verbose=True):
        assert(mode in ("Boolean", "Quantitative"))
        print("====================================================================================")
        print("Initializing MILP solver...")

        self.mode = mode
        self.verbose = verbose
        self.reset()

    def reset(self):
        # Initialize attributes
        self.objective = []
        self.node_vars = {}
        self.contract_vars = {}
        self.interval = []

        # Initialize convex solver
        self.model = gp.Model() # Convex solver for solving the MILP convex problem
        self.model.setParam("OutputFlag", 0)
        self.model.setParam("NonConvex", 2)

    def add_contract_variables(self, contract):
        """
        Adds a contract to the MILP solver.
        """
        [_, horizon_end_a] = contract.assumption.find_horizon(0, 0)
        [_, horizon_end_g] = contract.guarantee.find_horizon(0, 0)
        self.interval = [0, max(horizon_end_a, horizon_end_g)]
        for i, variable in enumerate(contract.deter_var_list):
            self.contract_vars[variable] = self.model.addVars(1, max(int(horizon_end_a), int(horizon_end_g))+1, lb=contract.deter_var_bounds[i][0], ub=contract.deter_var_bounds[i][1], vtype=GRB.BINARY if contract.deter_var_types == 'BINARY' else GRB.CONTINUOUS, name=variable)
        self.model.update()

    def set_objective(self, sense='minimize'):
        # print(self.soft_constraints)
        # print(self.soft_constraints_var)
        # input()
        if self.solver == "Gurobi":
            if sense == 'minimize':
                try:
                    self.model.remove(self.model.getConstrByName('UNSAT'))
                except:
                    pass
                self.model.addConstr(self.soft_constraints_var[0] <= -EPS, 'SAT')
            else:
                try:
                    self.model.remove(self.model.getConstrByName('SAT'))
                except:
                    pass
                self.model.addConstr(self.soft_constraints_var[0] >= 0, 'UNSAT')
            self.model.update()
        #  elif self.solver == "Cplex":
        #      self.model.minimize(objective)
        else: assert(False)

    def solve(self):
        """ Solves the MILP problem """
        # Solve the optimization problem
        self.model.write('MILP.lp')
        self.model.optimize()

        if self.status():
            if self.verbose:
                print("MILP solved.")
                self.print_solution()
            return True
        else:
            self.model.computeIIS()
            self.model.write("model.ilp")
            if self.verbose:
                print('There exists no solution.')
            return False

    def add_constraint(self, node, hard=True):
        """
        Adds contraints of a StSTL formula to the solvers.
        """
        # Build the parse tree
        processed_node = node.push_negation()
        processed_node.printInfo()

        # Add the constraints
        if self.mode == 'Boolean':
            self.node_vars['b'] = self.model.addVar(name='b', vtype=GRB.BINARY)
            if hard:
                self.model.addConstr(self.node_vars['b'] == 1)
            else:
                self.soft_constraints_var.append(self.node_vars['b'])
        elif self.mode == 'Quantitative':
            self.node_vars['b'] = self.model.addVar(name='b', lb=-M, ub=M, vtype=GRB.CONTINUOUS)
            if hard:
                self.model.addConstr(self.node_vars['b'] >= 1)
            else:
                self.soft_constraints_var.append(self.node_vars['b'])
        self.model.update()

        # Add node constraints
        self.add_node_constraint(node, top_node=True)

    def add_node_constraint(self, node, start_time = 0, end_time = 0, top_node=False):
        """
        Adds convex constraints from a contract to a convex solver.
        """
        print(node)
        if top_node:
            b_name = "b"
        else:
            b_name = "b_{}".format(len(self.node_vars))
            # 1. Add Boolean variables to the MILP convex solver
            if self.mode == 'Boolean':
                self.node_vars[b_name] = self.model.addVars(1, end_time+1, vtype=GRB.BINARY, name=b_name)
            elif self.mode == 'Quantitative':
                self.node_vars[b_name] = self.model.addVars(1, end_time+1, lb=-M, ub=M, vtype=GRB.CONTINUOUS, name=b_name)
            
            self.model.update()

        #  2. construct constraints according to the operation type
        if type(node) in (AP, stAP):
            self.add_ap_stap_constraint(node, self.node_vars[b_name], start_time, end_time)
        elif type(node) == temporal_unary: # Temporal unary operators
            self.add_temporal_unary_constraint(node, self.node_vars[b_name], start_time, end_time)
        elif type(node) == temporal_binary: # Temporal binary operator
            self.add_temporal_binary_constraint(node, self.node_vars[b_name], start_time, end_time)
        elif type(node) == nontemporal_unary: # Non-temporal unary operator
            self.add_nontemporal_unary_constraint(node, self.node_vars[b_name], start_time, end_time)
        elif type(node) == nontemporal_multinary: # Non-temporal multinary operator
            self.add_nontemporal_multinary_constraint(node, self.node_vars[b_name], start_time, end_time)
        elif type(node) == boolean:
            self.add_boolean_constraint(node, self.node_vars[b_name], start_time, end_time)
        
        return self.node_vars[b_name]

    def add_ap_stap_constraint(self, node, parent_vars, start_time = 0, end_time = 0):
        pass

        # #  1. construct the expression for constraint: expr <= 0
        # expr = node.expr.deter_data.copy()
        # constant = node.expr.constant
        # if node.ast_type == 'StAP': # If the current node is a stochastic AP,
        #     assert(node.expr.nondeter_data.shape[1] == 1)
        #     data = node.expr.nondeter_data[:, 0]
        #     data_size = node.expr.nondeter_data.shape[0]

        #     # Find expectation
        #     tmp_mean = self.contract.nondeter_var_mean[:data_size]
        #     expectation = 0
        #     for i in range(len(data)):
        #         expectation += data[i]*tmp_mean[i]

        #     # Find variance
        #     tmp_cov = self.contract.nondeter_var_cov[:data_size]
        #     tmp_cov = [tmp_cov[i][:data_size] for i in range(len(tmp_cov))]
        #     intermediate_variance = [0]*len(tmp_cov[0])
        #     variance = 0

        #     for i in range(len(data)):
        #         for j in range(len(tmp_cov[0])):
        #             for k in range(len(tmp_cov)):
        #                 intermediate_variance[i] += data[i]*tmp_cov[k][j]
            
        #     print(node)
        #     print(data)
        #     # print(data_size)
        #     print(intermediate_variance)

        #     for i in range(len(intermediate_variance)):
        #         variance += intermediate_variance[i]*data[i]
        #     # variance = data @ self.contract.nondeter_var_cov[:data_size, :data_size] @ data.T
        #     # print(intermediate_variance)
        #     print(expectation)
        #     print(variance)
        #     print(node.probability())

        #     if isinstance(node.probability(), float) or isinstance(variance, int):
        #         constant += expectation + norm.ppf(node.probability())*math.sqrt(variance)
        #     else:
        #         # print(node.probability())
        #         # print(self.contract.deter_var_list)
        #         # print(self.contract.deter_var_name2id)
        #         # Find lower and upper bounds for the probability threshold
        #         p_low = []
        #         p_high = []
        #         for deter_var in self.contract.deter_var_list:
        #             p_low.append(deter_var.bound[0])
        #             p_high.append(deter_var.bound[1])
        #         p_low = (p_low[0:len(node.probability().deter_data)]@node.probability().deter_data)[0]
        #         if p_low <= 0:
        #             p_low = 10**-16
        #         p_high = (p_high[0:len(node.probability().deter_data)]@node.probability().deter_data)[0]
        #         if p_high >= 1:
        #             p_high = 1- 10**-16
        #         # print((p_low, p_high))

        #         # Find the LHS of the inequality
        #         variance.sqrt = True
        #         if p_low >= 0.5:
        #             constant += expectation + (norm.ppf(p_low)+(norm.ppf(p_high)-norm.ppf(p_low))/(p_high-p_low)*(node.probability()-p_low))*variance
        #             # print(constant)
        #             # input()
        #         else:
        #             constant += expectation + (norm.ppf((p_low+p_high)/2) + 1/norm.pdf(norm.ppf((p_low+p_high)/2))*(node.probability()-(p_low+p_high)/2))*variance
        # variables = np.nonzero(expr)

        # #  2. add variables of contract in every concerned time period to model
        # for t in range(start_time, end_time):
        #     for v in variables[0]:
        #         if self.contract.deter_var_list[v].data_type == 'BINARY':
        #             self.model_add_binary_variable(v, t, var_type = 'contract')
        #         elif self.contract.deter_var_list[v].data_type == 'CONTINUOUS':
        #             lb = self.contract.deter_var_list[v].bound[0]
        #             ub = self.contract.deter_var_list[v].bound[1]
        #             self.model_add_continous_variable(v, t, var_type = 'contract', lb = lb, ub = ub)
        #         else: assert(False)
        #     if not isinstance(node.probability(), float):
        #         for v in np.nonzero(node.probability().deter_data)[0]:
        #             if self.contract.deter_var_list[v].data_type == 'BINARY':
        #                 self.model_add_binary_variable(v, t, var_type = 'contract')
        #             elif self.contract.deter_var_list[v].data_type == 'CONTINUOUS':
        #                 lb = self.contract.deter_var_list[v].bound[0]
        #                 ub = self.contract.deter_var_list[v].bound[1]
        #                 self.model_add_continous_variable(v, t, var_type = 'contract', lb = lb, ub = ub)
        #             else: assert(False)

        #     # 3. Add convex constraints
        #     self.model_add_inequality_constraint(node.idx, t, expr, constant)

    def add_temporal_unary_constraint(self, node, parent_vars, start_time = 0, end_time = 0):
        child_vars = self.add_node_constraint(node.children_list[0], start_time=start_time+node.interval[0], end_time=end_time+node.interval[1])

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

    def add_temporal_binary_constraint(self, node, parent_vars, start_time = 0, end_time = 0):
        child_vars1 = self.add_node_constraint(node.children_list[0], start_time=start_time, end_time=end_time+node.interval[1]-1)
        child_vars2 = self.add_node_constraint(node.children_list[1], start_time=start_time+node.interval[0], end_time=end_time+node.interval[1])

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

    def add_nontemporal_unary_constraint(self, node, parent_vars, start_time = 0, end_time = 0):
        # Fetch node variables from the child node
        child_vars = self.add_node_constraint(node.children_list[0], start_time=start_time, end_time=end_time+1)

        # Encode the logic 
        if type(parent_vars) == gp.Var:
            self.model.addConstr(parent_vars == child_vars[0, 1])
        else:
            for t in range(start_time, end_time):
                self.model.addConstr(parent_vars[0, t] == child_vars[0, t+1])
        self.model.update()

    def add_nontemporal_multinary_constraint(self, node, parent_vars, start_time = 0, end_time = 0):
        # Fetch node variables from the child node
        child_vars_list = []
        for children in node.children_list:
            child_vars_list.append(self.add_node_constraint(children, start_time=start_time, end_time=end_time))

        # Encode the logic 
        if type(parent_vars) == gp.Var:
            if node.operator == '&': # AND
                if self.mode == 'Boolean':
                    self.model.addConstr(parent_vars == gp.and_(child_vars_list))
                else:
                    self.model.addConstr(parent_vars == gp.min_(child_vars_list))
            elif node.ast_type == '|': # OR
                if self.mode == 'Boolean':
                    self.model.addConstr(parent_vars == gp.or_(child_vars_list))
                else:
                    self.model.addConstr(parent_vars == gp.max_(child_vars_list))
        else:
            for t in range(start_time, end_time):
                tmp_list = []
                for child_vars in child_vars_list:
                    tmp_list.append(child_vars[0,t])
                if node.operator == '&': # AND
                    if self.mode == 'Boolean':
                        self.model.addConstr(parent_vars[0,t] == gp.and_(tmp_list))
                    else:
                        self.model.addConstr(parent_vars[0,t] == gp.min_(tmp_list))
                elif node.ast_type == '|': # OR
                    if self.mode == 'Boolean':
                        self.model.addConstr(parent_vars[0,t] == gp.or_(tmp_list))
                    else:
                        self.model.addConstr(parent_vars[0,t] == gp.max_(tmp_list))

    def add_boolean_constraint(self, node, parent_vars, start_time = 0, end_time = 0):
        # print(node)
        pass

    def model_add_inequality_constraint(self, node_idx, time, expr, constant):
        # if (self.debug):
        print("adding inequality constraint: node id: {}, time: {}, expr: {}, constant: {}".format(node_idx, time, expr, constant))
        
        # Instialize the constraint
        constr = 0
        if isinstance(constant, float) or isinstance(constant, int):
            constr += constant
        else:
            constr += constant.constant
            var_idx = 0
            for num_list in constant.deter_data:
                constr += num_list[0]*self.contract_variable[var_idx, time]
                var_idx += 1
        
        #  1. handle linear term
        if len(expr) != 0:
            mask = np.nonzero(expr[:,0])[0]
            if mask.size > 0:
                # print(np.sum(self.contract_variable[mask, time] * expr[mask, 0]))
                constr += np.sum(self.contract_variable[mask, time] * expr[mask, 0])

            #  2. handle quadratic term
            if expr.shape[1] >= 2:
                mask = np.nonzero(expr[:,1])[0]
                for m in mask:
                    multiplier = expr[m, 1]
                    constr += multiplier * (self.contract_variable[m, time] * self.contract_variable[m, time])

        if self.debug:
            print("constraint: {}".format(constr <= 0))
            #  print("mask: {}".format(mask))
            #  print("variable: {}".format(variable))
            #  print("multiplier: {}".format(multiplier))
            #  print("rhs: {}".format(rhs))

        if self.mode == 'Boolean':
            if self.solver == "Gurobi":
                self.model.addConstr(M * (1 - self.node_variable[node_idx, time]) >= constr)
                self.model.addConstr(EPS - M * (self.node_variable[node_idx, time]) <= constr)
                # self.model.addConstr(constr >= - M * (self.node_variable[node_idx, time]))
            #  elif self.solver == "Cplex":
            #      lin_expr = [[[self.node_variable[node_idx, time]] + variable.tolist(), [M] + multiplier.tolist()]]
            #      self.model.linear_constraints.add(lin_expr = (lin_expr * 2), senses = "LG", rhs = [-rhs+M, -rhs+EPS])
            else: assert(False)
        elif self.mode == 'Quantitative':
            if self.solver == "Gurobi":
                self.model.addConstr(self.node_variable[node_idx, time] == -constr)
            #  elif self.solver == "Cplex":
            #      lin_expr = [[[self.node_variable[node_idx, time]] + variable.tolist(), [M] + multiplier.tolist()]]
            #      self.model.linear_constraints.add(lin_expr = lin_expr, senses = "E", rhs = [-rhs])
            else: assert(False)
        else: assert(False)

    def model_add_constraint(self, variable, multiplier, rhs, var_type = 'contract'):
        if (var_type == 'node'):
            variable_array = self.node_variable
        elif (var_type == 'contract'):
            variable_array = self.contract_variable
        else: assert(False)

        if (self.debug):
            print("adding constraint: variable: {}, multiplier: {}, rhs: {}".format(variable_array[variable[0], variable[1]], multiplier, rhs))

        if self.solver == "Gurobi":
            self.model.addConstr(np.sum(variable_array[variable[0], variable[1]] * multiplier) == rhs)
        elif self.solver == "Cplex":
            lin_expr = [[variable_array[variable[0], variable[1]].tolist(), multiplier.tolist()]]
            self.model.linear_constraints.add(lin_expr = lin_expr, senses = "E", rhs = [rhs])
        else: assert(False)

    def model_add_constraint_and(self, var, var_list):
        if (self.debug):
            print("adding and constraint: var: {}, expr: {}".format(var, var_list))

        if self.mode == 'Boolean':
            self.model.addConstr(var == gp.and_(var_list))

    def status(self):
        return self.model.getAttr("Status") == 2

    def print_solution(self, switching=False):
        # Print node variables
        for var_name, gurobi_vars in self.node_vars.items():
            if type(gurobi_vars) == gp.Var:
                print("{}: {}".format(var_name, gurobi_vars.x))
            else:
                for var_idx, gurobi_var in gurobi_vars.items():
                    print("{}[{}]: {}".format(var_name, var_idx[1], gurobi_var.x))

        # Print contract variables
        for var_name, gurobi_var in self.contract_vars.items():
            for t in range(int(self.interval[1])):
                print("{}[{}]: {}".format(var_name, t, gurobi_var[0, t].x))
        print()
    
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