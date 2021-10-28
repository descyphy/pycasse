# from copy import deepcopy
# import cplex
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
    __slots__ = ('model', 'contract', 'interval', 'node_vars', 'contract_vars', 'mode', 'objective', 'verbose', 'soft_constraint_vars')

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
        self.objective = []
        self.node_vars = {}
        self.contract_vars = {}
        self.interval = []
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

        # Add deterministic variables
        [_, horizon_end_a] = contract.assumption.find_horizon(0, 0)
        [_, horizon_end_g] = contract.guarantee.find_horizon(0, 0)
        self.interval = [0, max(horizon_end_a, horizon_end_g)]
        for i, variable in enumerate(contract.deter_var_list):
            self.contract_vars[variable] = self.model.addVars(1, max(int(horizon_end_a), int(horizon_end_g))+1, lb=contract.deter_var_bounds[i][0], ub=contract.deter_var_bounds[i][1], vtype=GRB.BINARY if contract.deter_var_types == 'BINARY' else GRB.CONTINUOUS, name=variable)
        self.model.update()

        # Add parametric variables
        for i, variable in enumerate(contract.param_var_list):
            self.model.addVar(lb=contract.param_var_bounds[i][0], ub=contract.param_var_bounds[i][1], vtype=GRB.CONTINUOUS, name=variable)
        self.model.update()

    def set_objective(self, sense='minimize'):
        # input()
        if sense == 'minimize':
            try:
                self.model.remove(self.model.getConstrByName('UNSAT'))
            except:
                pass
            if self.mode == 'Quantitative':
                self.model.addConstr(self.soft_constraint_vars[0] <= -EPS, 'SAT')
            else:
                self.model.addConstr(self.soft_constraint_vars[0] == 0, 'SAT')
        else:
            try:
                self.model.remove(self.model.getConstrByName('SAT'))
            except:
                pass
            if self.mode == 'Quantitative':
                self.model.addConstr(self.soft_constraint_vars[0] >= 0, 'UNSAT')
            else:
                self.model.addConstr(self.soft_constraint_vars[0] == 1, 'UNSAT')
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
        node.printInfo()
        processed_node = node.push_negation()
        processed_node.printInfo()

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
        if type(node) == AP: # Add constraints for an atomic predicate
            if type(parent_vars) == gp.Var:
                constr = 0
                for i in range(len(node.multipliers)):
                    term = node.multipliers[i]
                    for var_list in node.var_list_list[i]:
                        for var in var_list:
                            if var != 1:
                                term *= self.model.getVarByName("{}[0,0]".format(var))
                    constr += term
                if node.equal:
                    self.model.addConstr(parent_vars == constr)
                    self.model.addConstr(constr == 0)
                else:
                    if self.mode == 'Boolean':
                        self.model.addConstr(M * (1 - parent_vars) >= constr)
                        self.model.addConstr(EPS - M * parent_vars <= constr)
                    else:
                        self.model.addConstr(parent_vars == -constr)
            
            else:
                for t in range(start_time, end_time+1):
                    constr = 0
                    for i in range(len(node.multipliers)):
                        term = node.multipliers[i]
                        for var in node.var_list_list[i]:
                            if var != 1:
                                term *= self.model.getVarByName("{}[0,{}]".format(var, t))
                        constr += term
                    if node.equal:
                        self.model.addConstr(parent_vars[0,t] == constr)
                        self.model.addConstr(constr == 0)
                    else:
                        if self.mode == 'Boolean':
                            self.model.addConstr(M * (1 - parent_vars[0,t]) >= constr)
                            self.model.addConstr(EPS - M * parent_vars[0,t] <= constr)
                        else:
                            self.model.addConstr(parent_vars[0,t] == -constr)
        
        elif type(node) == stAP: # Add constraints for a stochastic atomic predicates
            # print(node)
            # Fetch the probability expression for the node
            prob = 0
            for i in range(len(node.prob_multipliers)):
                term = node.prob_multipliers[i]
                for var in node.prob_var_list_list[i]:
                    if var != 1:
                        term *= self.model.getVarByName(var)
                    prob += term

            if isinstance(prob, (float, int)): 
                # Handle extreme cases
                if prob == 0:
                    prob = 10**-16
                elif prob == 1:
                    prob = 1 - 10**-16
            else:
                # print(self.contract)
                # print(node)
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

                # print(p_low_mat)
                # print(p_high_mat)
                # print(p_multipliers)
                
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

            # Fetch mean and variance of stAP
            a = [0]*len(self.contract.nondeter_var_list)
            for i in range(len(node.multipliers)):
                for var in node.var_list_list[i]:
                    if var in self.contract.nondeter_var_list:
                        idx = self.contract.nondeter_var_list.index(var)
                        a[idx] = node.multipliers[i]
            # print(a)

            # Find mean
            mean1 = 0
            for i in range(len(a)):
                if isinstance(self.contract.nondeter_var_mean[i], (float, int)):
                    mean1 += a[i]*self.contract.nondeter_var_mean[i]
                else: # str
                    mean1 += a[i]*self.model.getVarByName(self.contract.nondeter_var_mean[i])
            # print(mean1)

            # Find variance
            # print(self.contract.nondeter_var_cov)
            intermediate_variance = [0]*len(a)
            for i in range(len(self.contract.nondeter_var_cov[0])):
                for j in range(len(a)):
                    if isinstance(self.contract.nondeter_var_cov[j][i], (float, int)):
                        intermediate_variance[i] += a[j]*self.contract.nondeter_var_cov[j][i]
                    else: # str
                        idx = self.contract.nondeter_var_cov[j][i].index('^')
                        intermediate_variance[i] += a[j]*self.model.getVarByName(self.contract.nondeter_var_cov[j][i][0:idx])*self.model.getVarByName(self.contract.nondeter_var_cov[j][i][0:idx])

            # print(intermediate_variance)

            variance = 0
            for i in range(len(intermediate_variance)):
                variance += intermediate_variance[i]*a[i]

            # Add deterministic variables to mean
            if type(parent_vars) == gp.Var:
                mean2 = 0
                for i in range(len(node.multipliers)):
                    term = node.multipliers[i]
                    for var in node.var_list_list[i]:
                        if var in self.contract.deter_var_list:
                            if var != 1 and 'w' not in var:
                                term *= self.model.getVarByName("{}[0,0]".format(var))
                        elif var in self.contract.param_var_list:
                            term *= self.model.getVarByName("{}".format(var))
                    if node.var_list_list[i] == [1] or any('w' not in var for var in node.var_list_list[i]):
                        mean2 += term
                
                # print(prob)
                # print(mean1+mean2)
                # print(variance)
                # input()

                # Find chance constraint
                if isinstance(prob, (float, int)) and isinstance(variance, (float, int)):
                    # Find LHS of the constraint 
                    constr = mean1 + mean2 + norm.ppf(prob)*math.sqrt(variance)
                elif isinstance(prob, (gp.QuadExpr, gp.LinExpr, gp.Var)) and isinstance(variance, (float, int)):
                    # Find LHS of the constraint 
                    if p_low >= 0.5:
                        constr = mean1 + mean2 + (norm.ppf(p_low)+(norm.ppf(p_high)-norm.ppf(p_low))/(p_high-p_low)*(prob-p_low))*math.sqrt(variance)
                    else:
                        constr = mean1 + mean2 + (norm.ppf((p_low+p_high)/2) + 1/norm.pdf(norm.ppf((p_low+p_high)/2))*(prob-(p_low+p_high)/2))*math.sqrt(variance)
                elif isinstance(prob, (float, int)) and isinstance(variance, (gp.QuadExpr, gp.LinExpr, gp.Var)):
                    # Add anxiliary variables and constraints
                    tmp_variance = self.model.addVar(name='{}_variance'.format(name), ub = 10**4, vtype=GRB.CONTINUOUS)
                    tmp_sigma = self.model.addVar(name='{}_sigma'.format(name), ub = 10**4, vtype=GRB.CONTINUOUS)
                    self.model.addConstr(tmp_variance == variance)
                    self.model.addGenConstrPow(tmp_variance, tmp_sigma, 0.5)
                    self.model.update()
                    
                    # Find LHS of the constraint 
                    constr = mean1 + mean2 + norm.ppf(prob)*tmp_sigma
                elif isinstance(prob, (gp.QuadExpr, gp.LinExpr, gp.Var)) and isinstance(variance, (gp.QuadExpr, gp.LinExpr, gp.Var)):
                    # Add anxiliary variables and constraints
                    tmp_variance = self.model.addVar(name='{}_variance'.format(name), ub = 10**4, vtype=GRB.CONTINUOUS)
                    tmp_sigma = self.model.addVar(name='{}_sigma'.format(name), ub = 10**4, vtype=GRB.CONTINUOUS)
                    self.model.addConstr(tmp_variance == variance)
                    self.model.addGenConstrPow(tmp_variance, tmp_sigma, 0.5)
                    self.model.update()
                    
                    # Find LHS of the constraint 
                    if p_low >= 0.5:
                        constr = mean1 + mean2 + (norm.ppf(p_low)+(norm.ppf(p_high)-norm.ppf(p_low))/(p_high-p_low)*(prob-p_low))*tmp_sigma
                    else:
                        constr = mean1 + mean2 + (norm.ppf((p_low+p_high)/2) + 1/norm.pdf(norm.ppf((p_low+p_high)/2))*(prob-(p_low+p_high)/2))*tmp_sigma
                
                # print(constr)

                # Add constraints
                if node.equal:
                    self.model.addConstr(parent_vars == constr)
                    self.model.addConstr(constr == 0)
                else:
                    if self.mode == 'Boolean':
                        self.model.addConstr(M * (1 - parent_vars) >= constr)
                        self.model.addConstr(EPS - M * parent_vars <= constr)
                    else:
                        self.model.addConstr(parent_vars == -constr)
                self.model.update()

            else:
                for t in range(start_time, end_time+1):
                    mean2 = 0
                    for i in range(len(node.multipliers)):
                        term = node.multipliers[i]
                        for var in node.var_list_list[i]:
                            if var in self.contract.deter_var_list:
                                if var != 1 and 'w' not in var:
                                    term *= self.model.getVarByName("{}[0,{}]".format(var, t))
                            elif var in self.contract.param_var_list:
                                term *= self.model.getVarByName("{}".format(var))
                        if node.var_list_list[i] == [1] or any('w' not in var for var in node.var_list_list[i]):
                            mean2 += term
                
                    # print(prob)
                    # input()
                    
                    # Find chance constraint
                    if isinstance(prob, (float, int)) and isinstance(variance, (float, int)):
                        # Find LHS of the constraint 
                        constr = mean1 + mean2 + norm.ppf(prob)*math.sqrt(variance)
                    elif isinstance(prob, (gp.QuadExpr, gp.LinExpr, gp.Var)) and isinstance(variance, (float, int)):
                        # Find LHS of the constraint 
                        if p_low >= 0.5:
                            con
                    # print(mean1+mean2)
                    # print(variance)str = mean1 + mean2 + (norm.ppf(p_low)+(norm.ppf(p_high)-norm.ppf(p_low))/(p_high-p_low)*(prob-p_low))*math.sqrt(variance)
                        else:
                            constr = mean1 + mean2 + (norm.ppf((p_low+p_high)/2) + 1/norm.pdf(norm.ppf((p_low+p_high)/2))*(prob-(p_low+p_high)/2))*math.sqrt(variance)
                    elif isinstance(prob, (float, int)) and isinstance(variance, (gp.QuadExpr, gp.LinExpr, gp.Var)):
                        # Add anxiliary variables and constraints
                        tmp_variance = self.model.addVar(name='{}_variance'.format(name), ub = 10**4, vtype=GRB.CONTINUOUS)
                        tmp_sigma = self.model.addVar(name='{}_sigma'.format(name), ub = 10**4, vtype=GRB.CONTINUOUS)
                        self.model.addConstr(tmp_variance == variance)
                        self.model.addGenConstrPow(tmp_variance, tmp_sigma, 0.5)
                        self.model.update()
                        
                        # Find LHS of the constraint 
                        constr = mean1 + mean2 + norm.ppf(prob)*tmp_sigma
                    elif isinstance(prob, (gp.QuadExpr, gp.LinExpr, gp.Var)) and isinstance(variance, (gp.QuadExpr, gp.LinExpr, gp.Var)):
                        # Add anxiliary variables and constraints
                        tmp_variance = self.model.addVar(name='{}_variance'.format(name), ub = 10**4, vtype=GRB.CONTINUOUS)
                        tmp_sigma = self.model.addVar(name='{}_sigma'.format(name), ub = 10**4, vtype=GRB.CONTINUOUS)
                        self.model.addConstr(tmp_variance == variance)
                        self.model.addGenConstrPow(tmp_variance, tmp_sigma, 0.5)
                        self.model.update()
                        
                        # Find LHS of the constraint 
                        if p_low >= 0.5:
                            constr = mean1 + mean2 + (norm.ppf(p_low)+(norm.ppf(p_high)-norm.ppf(p_low))/(p_high-p_low)*(prob-p_low))*tmp_sigma
                        else:
                            constr = mean1 + mean2 + (norm.ppf((p_low+p_high)/2) + 1/norm.pdf(norm.ppf((p_low+p_high)/2))*(prob-(p_low+p_high)/2))*tmp_sigma
                    
                    # print(constr)

                    # Add constraints
                    if node.equal:
                        self.model.addConstr(parent_vars[0,t] == constr)
                        self.model.addConstr(constr == 0)
                    else:
                        if self.mode == 'Boolean':
                            self.model.addConstr(M * (1 - parent_vars[0,t]) >= constr)
                            self.model.addConstr(EPS - M * parent_vars[0,t] <= constr)
                        else:
                            self.model.addConstr(parent_vars[0,t] == -constr)
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
                for t in range(start_time, end_time):
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
                for t in range(start_time, end_time):
                    if self.mode == 'Boolean':
                        self.model.addConstr(parent_vars[0,t] == 0)
                    else:
                        self.model.addConstr(parent_vars[0,t] == -M)

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
        for var_name, gurobi_var in self.contract_vars.items():
            for t in range(int(self.interval[1])+1):
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