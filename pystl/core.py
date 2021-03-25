import gurobipy as gp
import numpy as np

from gurobipy import GRB
from z3 import *
from pystl.contracts import *
from pystl.parser import *
from scipy.stats import norm

M = 10**4
EPS = 10**-4

class SMCSolver:
    """

    Attributes
        sat_solver
        main_convex_solver
        SSF_convex_solver
        b_vars
        b_var_num
    """

    def __init__(self, verbose=False, solver='gurobi'):
        print("====================================================================================")
        print("Initializing SMC solver...")

        # Initialize attributes
        self.contract = None
        self.dynamics = None
        self.verbose = verbose
        self.start_time = 0
        self.end_time = M
        self.prop_formula = ''

        # Initialize SAT solver
        self.sat_solver = Solver()
        self.sat_var = {}

        # Initialize convex solvers
        self.main_convex_solver = gp.Model() # Convex solver for solving the main convex problem
        self.SSF_convex_solver = gp.Model()  # Convex solver for solving the SSF convex problem

        if not verbose:
            self.main_convex_solver.setParam("OutputFlag", 0)
            self.SSF_convex_solver.setParam("OutputFlag", 0)
        
        # Initialize dictionary of Boolean variables that are shared between SAT and convex solvers
        self.bvar = {}

        # Initialize dictionary of Gurobi variables
        self.main_convex_var = {}
        self.SSF_convex_var = {}
        
        # Initialize dictionary of slack variables
        self.slack_var = {}

    def add_contract(self, contract):
        """
        Adds a contract to the SMC solver.
        """
        self.contract = contract

    def add_dynamics(self, dynamics):
        """
        Adds a dynamics to the SMC solver.
        """
        self.dynamics = dynamics

    def add_constraints(self):
        """
        Adds contraints to the solvers.
        """
        # Build the parse tree
        if self.contract.assumption == 'True':
            parse_tree_root, start_time, end_time = parse_ststl('b', self.contract.guarantee)
        else:
            parse_tree_root, start_time, end_time = parse_ststl('b', '(& ' + self.contract.assumption + ' ' + self.contract.guarantee + ')')
            # parse_tree_root, start_time, end_time = parse_ststl('b', self.contract.sat_guarantee)

        # Set the time horizon
        self.start_time = start_time
        self.end_time = end_time

        # Add the constraints from the dynamics
        if self.dynamics is not None:
            self.add_dyn_constraints()

        # Add the constraints from the parse tree
        self.add_contract_constraints(parse_tree_root)

    def add_contract_constraints(self, parse_tree_node, start_time = 0, end_time = 0):
        """
        Adds SAT and convex constraints from a contract to SAT and convex solvers.
        """
        
        # print(parse_tree_node)

        if parse_tree_node.name == 'b':
            self.prop_formula = parse_tree_node.name + '_t_0'

        if parse_tree_node.class_id in ('AP', 'StAP'): # If the current node is an AP or a StAP, add convex constraints
            
            # Add Boolean variables that are shared between SAT and convex solvers
            for i in range(parse_tree_node.start_time, parse_tree_node.end_time+1):
                self.sat_var[parse_tree_node.name + '_t_' + str(i)] = Bool(parse_tree_node.name + '_t_' + str(i))
                self.main_convex_var[parse_tree_node.name + '_t_' + str(i)] = self.main_convex_solver.addVar(vtype=GRB.BINARY, name=parse_tree_node.name + '_t_' + str(i))
                self.SSF_convex_var[parse_tree_node.name + '_t_' + str(i)] = self.SSF_convex_solver.addVar(vtype=GRB.BINARY, name=parse_tree_node.name + '_t_' + str(i))
            
            # Update convex solvers
            self.main_convex_solver.update()
            self.SSF_convex_solver.update()
            
            # Initialize dictionary of variables and fetch the operator
            var_dict = {}
            operator = ''

            if parse_tree_node.class_id == 'AP': # If the current node is an AP,
                left_terms = parse_tree_node.expr_left.terms
                right_terms = parse_tree_node.expr_right.terms
                operator = parse_tree_node.operator

            else: # If the current node is a stochastic AP,
                left_terms = parse_tree_node.ap.expr_left.terms
                right_terms = parse_tree_node.ap.expr_right.terms
                operator = parse_tree_node.ap.operator

            for term in left_terms:
                if term.variable in var_dict.keys():
                    var_dict[term.variable] += term.multiplier
                else:
                    var_dict[term.variable] = term.multiplier

            for term in right_terms:
                if term.variable in var_dict.keys():
                    var_dict[term.variable] -= term.multiplier
                else:
                    var_dict[term.variable] = -term.multiplier
                
            if operator == "=>":
                for k, v in var_dict.items():
                    var_dict[k] = -v
            elif operator == '>':
                for k, v in var_dict.items():
                    var_dict[k] = -v
                var_dict[None] += EPS
            elif operator == '<':
                var_dict[None] += EPS

            for t in range(parse_tree_node.start_time, parse_tree_node.end_time+1):
                # Add slack variables
                slack_var_name = 's_{}_t_{}'.format(parse_tree_node.name, t)
                slack_anx_var_name = 'a_{}_t_{}'.format(parse_tree_node.name, t)
                self.main_convex_var[slack_var_name] = self.main_convex_solver.addVar(lb=-M, ub=M, vtype=GRB.CONTINUOUS, name=slack_var_name)
                self.main_convex_var[slack_anx_var_name] = self.main_convex_solver.addVar(lb=0, ub=M, vtype=GRB.CONTINUOUS, name=slack_anx_var_name)
                self.slack_var[slack_var_name] = M
                self.main_convex_solver.addConstr(self.main_convex_var[slack_anx_var_name] == gp.abs_(self.main_convex_var[slack_var_name]))
                
                self.main_convex_solver.update()

                # Add convex variables
                if self.dynamics is None:
                    for var in var_dict.keys():
                        if var is not None:
                            var_name = '{}_{}'.format(var, t)
                            if self.main_convex_solver.getVarByName(var_name) is None:
                                self.main_convex_var[var_name] = self.main_convex_solver.addVar(vtype=GRB.CONTINUOUS, lb = -M, ub = M, name = var_name)
                            
                            if self.SSF_convex_solver.getVarByName(var_name) is None:
                                self.SSF_convex_var[var_name] = self.SSF_convex_solver.addVar(vtype=GRB.CONTINUOUS, lb = -M, ub = M, name = var_name)

                    # Update convex solvers
                    self.main_convex_solver.update()
                    self.SSF_convex_solver.update()

            # Add convex constraints
            for t in range(parse_tree_node.start_time, parse_tree_node.end_time+1):
                tmp_main_const = 0
                tmp_SSF_const = 0
                for k, v in var_dict.items():
                    if k is not None:
                        tmp_main_const += v*self.main_convex_var[k+'_'+str(t)]
                        tmp_SSF_const += v*self.SSF_convex_var[k+'_'+str(t)]
                    else:
                        tmp_main_const += v
                        tmp_SSF_const += v
                self.main_convex_solver.addConstr((self.main_convex_var[parse_tree_node.name+'_t_'+str(t)] == 1) >> (tmp_main_const <= self.main_convex_var['s_'+parse_tree_node.name+'_t_'+str(t)]))
                self.main_convex_solver.addConstr((self.main_convex_var[parse_tree_node.name+'_t_'+str(t)] == 0) >> (tmp_main_const >= EPS + self.main_convex_var['s_'+parse_tree_node.name+'_t_'+str(t)]))
                self.SSF_convex_solver.addConstr((self.SSF_convex_var[parse_tree_node.name+'_t_'+str(t)] == 1) >> (tmp_SSF_const <= 0))
                self.SSF_convex_solver.addConstr((self.SSF_convex_var[parse_tree_node.name+'_t_'+str(t)] == 0) >> (tmp_SSF_const >= EPS))
            
            # Update the convex solvers
            self.main_convex_solver.update()
            self.SSF_convex_solver.update()

        else: # If the current node is not an AP or a StAP, add SAT constraints
            if parse_tree_node.operator in ('G', 'F'): # Temporal unary operators
                # Fetch a sub-formula
                subformula = parse_tree_node.formula_list[0]
                
                # Build tmp_prop_formula to encode the logic
                for i in range(start_time, end_time+1):
                    if parse_tree_node.operator == 'G':
                        tmp_prop_formula = '(and'
                    else:
                        tmp_prop_formula = '(or'
                    for j in range(parse_tree_node.start_time, parse_tree_node.end_time+1):
                        tmp_prop_formula += ' ' + subformula.name + '_t_' + str(i+j)
                    tmp_prop_formula += ')'
                    
                    # Replace a part of the current prop_formula with tmp_prop_formula
                    if parse_tree_node.name == 'b':
                        self.prop_formula = self.prop_formula.replace(parse_tree_node.name + '_t_' + str(i), tmp_prop_formula)
                    else:
                        self.prop_formula = self.prop_formula.replace(parse_tree_node.name + '_t_' + str(i) + ' ', tmp_prop_formula + ' ')
                        self.prop_formula = self.prop_formula.replace(parse_tree_node.name + '_t_' + str(i) + ')', tmp_prop_formula + ')')
                
                self.add_contract_constraints(subformula, start_time=start_time+parse_tree_node.start_time, end_time=end_time+parse_tree_node.end_time)
            
            elif parse_tree_node.operator in ('U', 'R'): # Temporal binary operator
                # Fetch two sub-formulas
                subformula1 = parse_tree_node.formula_list[0]
                subformula2 = parse_tree_node.formula_list[1]
                
                # Build tmp_prop_formula to encode the logic
                for i in range(start_time, end_time+1):
                    if parse_tree_node.operator == 'U':
                        tmp_prop_formula = '(or'
                        for j in range(parse_tree_node.start_time, parse_tree_node.end_time+1):
                            tmp_prop_formula += ' (and ' + subformula2.name + '_t_' + str(i+j)
                            for l in range(j):
                                tmp_prop_formula += ' ' + subformula1.name + '_t_' + str(i+l)
                            tmp_prop_formula += ')'
                    else:
                        tmp_prop_formula = '(and'
                        for j in range(parse_tree_node.start_time, parse_tree_node.end_time+1):
                            tmp_prop_formula += ' (or ' + subformula2.name + '_t_' + str(i+j)
                            for l in range(j):
                                tmp_prop_formula += ' ' + subformula1.name + '_t_' + str(i+l)
                            tmp_prop_formula += ')'
                    tmp_prop_formula += ')'
                    
                    # Replace a part of the current prop_formula with tmp_prop_formula
                    if parse_tree_node.name == 'b':
                        self.prop_formula = self.prop_formula.replace(parse_tree_node.name + '_t_' + str(i), tmp_prop_formula)
                    else:
                        self.prop_formula = self.prop_formula.replace(parse_tree_node.name + '_t_' + str(i) + ' ', tmp_prop_formula + ' ')
                        self.prop_formula = self.prop_formula.replace(parse_tree_node.name + '_t_' + str(i) + ')', tmp_prop_formula + ')')
            
                self.add_contract_constraints(subformula1, start_time=start_time+parse_tree_node.start_time, end_time=end_time+parse_tree_node.end_time)
                self.add_contract_constraints(subformula2, start_time=start_time+parse_tree_node.start_time, end_time=end_time+parse_tree_node.end_time)
            
            elif parse_tree_node.operator == '!': # Non-temporal unary operator
                # Fetch a sub-formula
                subformula = parse_tree_node.formula_list[0]
                
                # Build tmp_prop_formula to encode the logic
                for i in range(start_time, end_time+1):
                    tmp_prop_formula = '(not '+ subformula.name + '_t_' + str(i) + ")"
                    
                    # Replace a part of the current prop_formula with tmp_prop_formula
                    if parse_tree_node.name == 'b':
                        self.prop_formula = self.prop_formula.replace(parse_tree_node.name + '_t_' + str(i), tmp_prop_formula)
                    else:
                        self.prop_formula = self.prop_formula.replace(parse_tree_node.name + '_t_' + str(i) + ' ', tmp_prop_formula + ' ')
                        self.prop_formula = self.prop_formula.replace(parse_tree_node.name + '_t_' + str(i) + ')', tmp_prop_formula + ')')
                
                self.add_contract_constraints(subformula, start_time=start_time, end_time=end_time)

            else: # Non-temporal multinary operator
                # Build tmp_prop_formula to encode the logic
                for i in range(start_time, end_time+1):
                    if parse_tree_node.operator == '&':
                        tmp_prop_formula = '(and'
                        for subformula in parse_tree_node.formula_list:
                            tmp_prop_formula += ' ' + subformula.name + '_t_' + str(i)
                        tmp_prop_formula += ')'
                    else:
                        tmp_prop_formula = '(or'
                        for subformula in parse_tree_node.formula_list:
                            tmp_prop_formula += ' ' + subformula.name + '_t_' + str(i)
                        tmp_prop_formula += ')'

                    # Replace a part of the current prop_formula with tmp_prop_formula
                    if parse_tree_node.name == 'b':
                        self.prop_formula = self.prop_formula.replace(parse_tree_node.name + '_t_' + str(i), tmp_prop_formula)
                    else:
                        self.prop_formula = self.prop_formula.replace(parse_tree_node.name + '_t_' + str(i) + ' ', tmp_prop_formula + ' ')
                        self.prop_formula = self.prop_formula.replace(parse_tree_node.name + '_t_' + str(i) + ')', tmp_prop_formula + ')')
            
                for subformula in parse_tree_node.formula_list:
                    self.add_contract_constraints(subformula, start_time=start_time, end_time=end_time)

        if parse_tree_node.name == 'b': # Add SAT constraints
            tmp_prop_formula = "(assert " + self.prop_formula + ")"
            tmp_prop_formula = parse_smt2_string(tmp_prop_formula, decls = self.sat_var)
            self.sat_solver.add(tmp_prop_formula)

    def add_dyn_constraints(self):
        """
        Adds constraints for the system dynamics given as a discrete-time state-space model with process and measurement noise to the main convex solver.
        """
        # Add convex variables
        for t in range(self.start_time, self.end_time+1):
            for i in range(self.dynamics.x_len):
                var_name = 'x[{}]_{}'.format(i, t)
                lb = self.dynamics.x_bounds[i,0]
                ub = self.dynamics.x_bounds[i,1]
                self.main_convex_var[var_name] = self.main_convex_solver.addVar(vtype=GRB.CONTINUOUS, lb = lb, ub = ub, name = var_name)
                self.SSF_convex_var[var_name] = self.SSF_convex_solver.addVar(vtype=GRB.CONTINUOUS, lb = lb, ub = ub, name = var_name)
            
        for t in range(self.start_time, self.end_time+1):
            if self.dynamics.y_len is not None:
                for i in range(self.dynamics.y_len):
                    var_name = 'y[{}]_{}'.format(i, t)
                    lb = self.dynamics.y_bounds[i,0]
                    ub = self.dynamics.y_bounds[i,1]
                    self.main_convex_var[var_name] = self.main_convex_solver.addVar(vtype=GRB.CONTINUOUS, lb = lb, ub = ub, name = var_name)
                    self.SSF_convex_var[var_name] = self.SSF_convex_solver.addVar(vtype=GRB.CONTINUOUS, lb = lb, ub = ub, name = var_name)
        
        for t in range(self.start_time, self.end_time):
            if self.dynamics.u_len is not None:
                for i in range(self.dynamics.u_len):
                    var_name = 'u[{}]_{}'.format(i, t)
                    lb = self.dynamics.u_bounds[i,0]
                    ub = self.dynamics.u_bounds[i,1]
                    self.main_convex_var[var_name] = self.main_convex_solver.addVar(vtype=GRB.CONTINUOUS, lb = lb, ub = ub, name = var_name)
                    self.SSF_convex_var[var_name] = self.SSF_convex_solver.addVar(vtype=GRB.CONTINUOUS, lb = lb, ub = ub, name = var_name)

        # Update convex solvers 
        self.main_convex_solver.update()
        self.SSF_convex_solver.update()

        # Add the initial states constraints to convex solver
        for i in range(self.dynamics.x_len):
            self.main_convex_solver.addConstr(self.main_convex_var['x[' + str(i) + ']_0'] == self.dynamics.x0[i, 0])
            self.SSF_convex_solver.addConstr(self.SSF_convex_var['x[' + str(i) + ']_0'] == self.dynamics.x0[i, 0])

        # Update convex solvers 
        self.main_convex_solver.update()
        self.SSF_convex_solver.update()

        # Add convex constraints
        for t in range(self.start_time, self.end_time):
            self.main_convex_solver.addConstrs(self.main_convex_var['x[' + str(i) + ']_' + str(t+1)] \
                                                == gp.quicksum(self.dynamics.A[i,j]*self.main_convex_var['x[' + str(j) + ']_' + str(t)] for j in range(self.dynamics.x_len)) \
                                                    + gp.quicksum(self.dynamics.B[i,l]*self.main_convex_var['u[' + str(l) + ']_' + str(t)] for l in range(self.dynamics.u_len)) for i in range(self.dynamics.x_len))
            self.SSF_convex_solver.addConstrs(self.SSF_convex_var['x[' + str(i) + ']_' + str(t+1)] \
                                                == gp.quicksum(self.dynamics.A[i,j]*self.SSF_convex_var['x[' + str(j) + ']_' + str(t)] for j in range(self.dynamics.x_len)) \
                                                    + gp.quicksum(self.dynamics.B[i,l]*self.SSF_convex_var['u[' + str(l) + ']_' + str(t)] for l in range(self.dynamics.u_len)) for i in range(self.dynamics.x_len))
            
            try:
                self.main_convex_solver.addConstrs(self.main_convex_var['y[' + str(i) + ']_' + str(t+1)] \
                                                    == gp.quicksum(self.dynamics.C[i,j]*self.main_convex_var['x[' + str(j) + ']_' + str(t)] for j in range(self.dynamics.x_len)) \
                                                        + gp.quicksum(self.dynamics.D[i,l]*self.main_convex_var['u[' + str(l) + ']_' + str(t)] for l in range(self.dynamics.u_len)) for i in range(self.dynamics.y_len))
                self.SSF_convex_solver.addConstrs(self.SSF_convex_var['y[' + str(i) + ']_' + str(t+1)] \
                                                    == gp.quicksum(self.dynamics.C[i,j]*self.SSF_convex_var['x[' + str(j) + ']_' + str(t)] for j in range(self.dynamics.x_len)) \
                                                        + gp.quicksum(self.dynamics.D[i,l]*self.SSF_convex_var['u[' + str(l) + ']_' + str(t)] for l in range(self.dynamics.u_len)) for i in range(self.dynamics.y_len))
            except:
                pass

        # Update convex solvers 
        self.main_convex_solver.update()
        self.SSF_convex_solver.update()

    # def add_NN_constraints(self):
    #   """
    #   Adds constraints for the NN to the main convex solver.
    #   """

    def solve(self, certificate='SSF'):
        """ Solves the SMC problem """
        # Add constraints of the contract and dynamics to the SAT, main, and SSF convex solver
        self.add_constraints()

        # Initialize variables
        solved = True
        count = 1

        # SMC
        while True:
            solved = self.solve_SAT()
            if solved:
                self.add_Bool_constraints()
                solved = self.solve_main_convex()
                # print(self.SSF)
                if solved:
                    if self.verbose:
                        print("SMC problem solved within {} iteration(s).\n".format(count))
                    return True
                else:
                    conflict_assignments = self.find_conflicts(certificate)
                    self.add_SAT_constraints(conflict_assignments)
            else:
                if self.verbose:
                    print('There exists no solution.\n')
                return False
            
            count += 1

    def solve_SAT(self):
        """
        Solves SAT problem.
        """
        # Solve the SAT problem and fetch the solution
        solved = self.sat_solver.check()

        if str(solved) == 'sat':
            solution = self.sat_solver.model()
            for var in solution:
                self.bvar[str(var())] = solution[var]
            return True
        else:
            # print(self.sat_solver)
            return False

    def add_Bool_constraints(self):
        """
        Adds Boolean constraints to the convex solver.
        """
        # Delete previous SAT assignments
        try:
            constraints = self.main_convex_solver.getConstrs()
            for constraint in constraints:
                if constraint.getAttr('ConstrName') == "b_assign":
                    self.main_convex_solver.remove(constraint)
        except:
            pass
        
        # Add constraints for Boolean assignments
        for k, v in self.bvar.items():
            if v:
                self.main_convex_solver.addConstr(self.main_convex_var[str(k)] == 1, name = "b_assign")
            else:
                self.main_convex_solver.addConstr(self.main_convex_var[str(k)] == 0, name = "b_assign")
        self.main_convex_solver.update()
        
    def solve_main_convex(self):
        """
        Solves main convex problem.
        """
        # Add slack objective
        obj = 0
        for v in self.main_convex_var.keys():
            if 'a_' in str(v):
                obj = obj + self.main_convex_var[v]
        self.main_convex_solver.setObjective(obj, GRB.MINIMIZE)
        self.main_convex_solver.update()

        # Solve the convex problem
        self.main_convex_solver.optimize()
        # self.main_convex_solver.write('main.lp')
        if self.main_convex_solver.getAttr("Status") == 2:
            # Update values of slack variables
            for v in self.main_convex_solver.getVars():
                # print('%s %g' % (v.varName, v.x))
                if 's_b_' in v.varName:
                    self.slack_var[v.varName] = v.x
                # if 'x' in v.varName or 'u' in v.varName:
                #   print('%s %g' % (v.varName, v.x))
            
            # Update SSF
            self.SSF = self.main_convex_solver.getObjective().getValue()
            if self.SSF < EPS:
                return True
            else:
                return False
        else:
            return False

    def find_conflicts(self, certificate):
        """
        Find conflicting assignments associated with the lowest slack.
        """
        # Initialize the conflict_assignments
        conflicts = []

        # Delete previous SAT assignments
        try:
            constraints = self.SSF_convex_solver.getConstrs()
            for constraint in constraints:
                if constraint.getAttr('ConstrName') == "b_assign":
                    self.SSF_convex_solver.remove(constraint)
        except:
            pass
        
        if certificate == 'IIS':
            # Add constraints for a Boolean assignment
            for bvar_name in self.bvar.keys():
                if self.bvar[bvar_name]:
                    self.SSF_convex_solver.addConstr(self.SSF_convex_var[bvar_name] == 1, name = "b_assign")
                else:
                    self.SSF_convex_solver.addConstr(self.SSF_convex_var[bvar_name] == 0, name = "b_assign")

            # Find IIS
            # self.SSF_convex_solver.write("SSF.lp")
            if self.SSF_convex_solver.getAttr("Status") != 2:
                self.SSF_convex_solver.computeIIS()
                self.SSF_convex_solver.write("IIS.ilp")

                # Read IIS file
                p = {"OutputFlag": 0}
                with gp.Env(params=p) as env, gp.read("IIS.ilp", env=env) as model:
                    # Find IIS Boolean variables
                    for v in model.getVars():
                        if 'b' in v.varName:
                            conflicts.append(v.varName)
                
            return conflicts

        elif certificate == 'SSF':
            # Sort the slacks
            count = 0
            self.slack_var = {k: v for k, v in sorted(self.slack_var.items(), key=lambda item: item[1], reverse=True)}
            lowest_slack_name = list(self.slack_var.keys())[-1]
            tmp_slack_name = list(self.slack_var.keys())[count]

            # Add constraints for a Boolean assignment
            bvar_name = lowest_slack_name[2:len(lowest_slack_name)]
            conflicts.append(bvar_name)
            if self.bvar[bvar_name]:
                self.SSF_convex_solver.addConstr(self.SSF_convex_var[bvar_name] == 1, name = "b_assign")
            else:
                self.SSF_convex_solver.addConstr(self.SSF_convex_var[bvar_name] == 0, name = "b_assign")
            self.SSF_convex_solver.update()

            # Solve the convex problem
            while True:
                # Add constraints for a Boolean assignment
                tmp_slack_name = list(self.slack_var.keys())[count]
                bvar_name = tmp_slack_name[2:len(tmp_slack_name)]
                conflicts.append(bvar_name)
                if self.bvar[bvar_name]:
                    self.SSF_convex_solver.addConstr(self.SSF_convex_var[bvar_name] == 1, name = "b_assign")
                else:
                    self.SSF_convex_solver.addConstr(self.SSF_convex_var[bvar_name] == 0, name = "b_assign")
                self.SSF_convex_solver.update()

                # Solve the convex problem
                self.SSF_convex_solver.optimize()
                # self.SSF_convex_solver.write('SSF.lp')

                if self.SSF_convex_solver.getAttr("Status") == 2:
                    count += 1
                else:
                    # self.SSF_convex_solver.computeIIS()
                    # self.SSF_convex_solver.write("IIS.ilp")
                    return conflicts

    def add_SAT_constraints(self, conflict_assignments):
        """
        Find conflicting assignments associated with the lowest slack.
        """
        # Add new SAT constraints
        block = []
        for bvar in conflict_assignments:
            block.append(self.sat_var[bvar] != self.bvar[bvar])
        self.sat_solver.add(Or(block))


class MILPSolver:
    """

    Attributes
        MILP_convex_solver
        b_vars
        b_var_num
    """

    def __init__(self, verbose=False, mode = "Boolean", solver = "Gurobi"):
        assert(mode in ("Boolean", "Quantitative"))
        assert(solver in ("Gurobi", "Cplex"))
        print("====================================================================================")
        print("Initializing MILP solver...")

        self.verbose = verbose
        self.mode = mode
        self.solver = solver
        self.objective = None

        self.reset()

    def reset(self):
        # Initialize attributes
        self.contract = None
        self.constraint = []

        # Initialize convex solvers
        if self.solver == "Gurobi":
            self.model = gp.Model() # Convex solver for solving the MILP convex problem
            if not self.verbose:
                self.model.setParam("OutputFlag", 0)
        elif self.solver == "Cplex":
            self.model = cpx.Model(name="Model") # Convex solver for solving the MILP convex problem
        else: assert(False)

        # Initialize dictionary of Gurobi variables
        self.variable = []
    
    def add_contract(self, contract):
        """
        Adds a contract to the SMC solver.
        """
        self.contract = contract

        if self.contract.assumption == 'True' and self.contract.guarantee == 'True':
            pass
        elif self.contract.assumption == 'True':
            self.constraint.append(self.contract.guarantee)
        else:
            self.constraint.append(self.contract.sat_guarantee)

    #  def add_dynamics(self, dynamics):
    #      """
    #      Adds a dynamics to the SMC solver.
    #      """
    #      #  self.dynamics = dynamics

    #  def add_constraint(self, constraint):
    #      """
    #      Adds a constraint to the SMC solver.
    #      """
    #      self.constraint.append(constraint)

    def solve(self, objective=None):
        """ Solves the MILP problem """
        # Add constraints of the contract and dynamics to the SAT, main, and SSF convex solver
        for c in self.constraint:
            self.set_constraint(c)


        # Solve the convex problem
        if self.solver == "Gurobi":
            self.model.optimize()
            self.model.write('MILP.lp')
        elif self.solver == "Cplex":
        else: assert(False)

        
        if self.model.getAttr("Status") == 2: # If MILP is successfully solved,
            if self.verbose:
                print("MILP solved.")
            # # Print the MILP solution
            # for v in self.model.getVars():
            #   if 'b' not in v.varName:
            #       print('%s %g' % (v.varName, v.x))
            return True

        else:
            if self.verbose:
                print('There exists no solution.')
            return False

    def set_constraint(self, constraint):
        """
        Adds contraints to the solvers.
        """
        # Build the parse tree
        parse_tree_root = Parser()(constraint)

        self.set_node_constraint(parse_tree_root)

        if self.mode == 'Boolean':
            if self.solver == "Gurobi":
                self.model.addConstr(self.variable[0][parse_tree_root.name] == 1)
            elif self.solver == "Cplex":
                self.model.add_constraint(self.variable[0][parse_tree_root.name] == 1)
            else: assert(False)
        elif self.mode == 'Quantitative':
            if self.solver == "Gurobi":
                self.model.addConstr(self.variable[0][parse_tree_root.name] >= 1)
            elif self.solver == "Cplex":
                self.model.add_constraint(self.variable[0][parse_tree_root.name] >= 1)
            else: assert(False)

        if self.objective == 'min':
            self.model.setObjective(self.MILP_convex_var['b_t_0'], GRB.MINIMIZE)
        elif self.objective == 'max':
            self.model.setObjective(self.MILP_convex_var['b_t_0'], GRB.MAXIMIZE)
    def set_node_constraint(self, node, start_time = 0, end_time = 1):
        """
        Adds SAT and convex constraints from a contract to SAT and convex solvers.
        """
        print(repr(node))
        input()

        # 1. Add Boolean variables to the MILP convex solver
        for t in range(start_time, end_time):
            if self.mode == 'Boolean':
                self.add_binary_variable(node.name, t)
            elif self.mode == 'Quantitative':
                self.add_continous_variable(node.name, t)
            else:
                assert(false)
        input()

        #  2. construct constraints according to the operation type
        if node.class_id in ('AP', 'StAP'):
            self.set_ap_stap_constraint(node, start_time, end_time)
        elif node.operator in ('G', 'F'): # Temporal unary operators
            self.set_G_F_constraint(node, start_time, end_time)
        elif node.operator in ('U', 'R'): # Temporal binary operator
            self.set_U_R_constraint(node, start_time, end_time)
        elif node.operator == '!': # Non-temporal unary operator
            assert(False)
        else: # Non-temporal multinary operator
            # Recursions
            for subformula in node.formula_list:
                self.set_node_constraint(subformula, start_time=start_time, end_time=end_time)

            # Add Boolean constraints to MILP convex solver
            for t in range(start_time, end_time):
                if node.operator == '&': # AND
                    self.add_constraint_and(self.variable[t][node.name], [self.variable[t][f.name] for f in node.formula_list])
                elif node.operator == '|': # OR
                    self.add_constraint_or(self.variable[t][node.name], [self.variable[t][f.name] for f in node.formula_list])
                else:
                    assert(False)
            input()
    def set_ap_stap_constraint(self, node, start_time = 0, end_time = 1):
        #  1. construct the expression for constraint: expr <= 0
        if node.class_id == 'AP': # If the current node is an AP,
            expr = node.expr
            operator = node.operator
        elif node.class_id == 'StAP': # If the current node is a stochastic AP,
            expr = node.ap.expr
            operator = node.ap.operator
        else: assert(False)
                
        if operator == "=>":
            expr *= -1
        elif operator == '>':
            expr *= -1
            expr.add(None, EPS)
        elif operator == '<':
            expr.add(None, EPS)
        else: assert(operator == '=>')

        print(expr)
        input()

        #  2. add variables of contract in every concerned time period to model
        #  3. caculate the usage of nondeterminate uncontrolled variables
        nondeter_uncontrolled_vars_term = np.zeros(len(self.contract.nondeter_uncontrolled_vars['var_names']))
        for (var_name , multiplier) in expr.terms.items():
            if var_name in self.contract.controlled_vars['var_names']:
                idx = self.contract.controlled_vars['var_names'].index(var_name)
                lb = self.contract.controlled_vars['bounds'][idx, 0]
                ub = self.contract.controlled_vars['bounds'][idx, 1]
            elif var_name in self.contract.deter_uncontrolled_vars['var_names']:
                idx = self.contract.deter_uncontrolled_vars['var_names'].index(var_name)
                lb = self.contract.deter_uncontrolled_vars['bounds'][idx, 0]
                ub = self.contract.deter_uncontrolled_vars['bounds'][idx, 1]
            elif var_name in self.contract.nondeter_uncontrolled_vars['var_names']:
                idx = self.contract.nondeter_uncontrolled_vars['var_names'].index(var_name)
                nondeter_uncontrolled_vars_term[idx] = multiplier
            else: assert(var_name == None)

            if var_name in self.contract.controlled_vars['var_names'] or var_name in self.contract.deter_uncontrolled_vars['var_names']:
                for t in range(start_time, end_time):
                    self.add_continous_variable(var_name, t, lb, ub)
        print(nondeter_uncontrolled_vars_term)
        input()

            
        # 4. Add convex constraints
        for t in range(start_time, end_time):
            constraint = 0
            for var_name, multiplier in expr.terms.items():
                if var_name in self.contract.controlled_vars['var_names'] or var_name in self.contract.deter_uncontrolled_vars['var_names']:
                    constraint += multiplier * self.variable[t][var_name]
                elif var_name == None:
                    constraint += multiplier
                else: assert(var_name in self.contract.nondeter_uncontrolled_vars['var_names'])

                # Add convex constraints
            if node.class_id == 'AP': # 'AP'
                self.add_constraint(self.variable[t][node.name], constraint)

            elif node.class_id == 'StAP': # 'StAP'
                expectation = nondeter_uncontrolled_vars_term@self.contract.nondeter_uncontrolled_vars['mean']
                variance = nondeter_uncontrolled_vars_term@self.contract.nondeter_uncontrolled_vars['cov']@nondeter_uncontrolled_vars_term.T

                constraint += expectation + norm.ppf(node.prob) * math.sqrt(variance)

                if (node.negation):
                    constraint *= -1

                self.add_constraint(self.variable[t][node.name], constraint)
        input()
    def set_G_F_constraint(self, node, start_time = 0, end_time = 1):
        self.set_node_constraint(node.formula_list[0], start_time=start_time+node.start_time, end_time=end_time+node.end_time)
        # Build tmp_prop_formula to encode the logic
        for t in range(start_time, end_time):
            if node.operator == 'G': # Globally
                self.add_constraint_and(self.variable[t][node.name], [self.variable[t_i][node.formula_list[0].name] for t_i in range(t + node.start_time, t + node.end_time + 1)])
            elif node.operator == 'F': # Finally
                self.add_constraint_or(self.variable[t][node.name], [self.variable[t_i][node.formula_list[0].name] for t_i in range(t + node.start_time, t + node.end_time + 1)])
        input()
    def set_U_R_constraint(self, node, start_time = 0, end_time = 1):
        self.set_node_constraint(node.formula_list[0], start_time=start_time, end_time=end_time+node.end_time)
        self.set_node_constraint(node.formula_list[1], start_time=start_time+node.start_time, end_time=end_time+node.end_time)

        for t in range(start_time + node.start_time, end_time + node.end_time):
            if self.mode == 'Boolean':
                self.add_binary_variable("{}_aux".format(node.name), t)
            elif self.mode == 'Quantitative':
                self.add_continous_variable("{}_aux".format(node.name), t)
            else:
                assert(false)
            if node.operator == 'U': # Globally
                self.add_constraint_and(self.variable[t]["{}_aux".format(node.name)], [self.variable[t_i][node.formula_list[0].name] for t_i in range(start_time, t)] + [self.variable[t][node.formula_list[1].name]])

        for t in range(start_time, end_time):
            if node.operator == 'U': # Globally
                # Add anxiliary Boolean constraints to MILP convex solver
                self.add_constraint_and(self.variable[t][node.name], [self.variable[t_i]["{}_aux".format(node.name)] for t_i in range(t + node.start_time, t + node.end_time)])
    def add_binary_variable(self, var_name, time):
        print("adding binary variable: var: {}, t: {}".format(var_name, time))
        self.variable.extend([{} for i in range(time + 1 - len(self.variable))])

        if self.solver == "Gurobi":
            self.variable[time][var_name] = self.model.addVar(vtype=GRB.BINARY, name="{}_{}".format(var_name, time))
            self.model.update()
        elif self.solver == "Cplex":
            self.variable[time][var_name] = self.model.binary_var(name="{}_{}".format(var_name, time))
        else: assert(False)
    def add_continous_variable(self, var_name, time, lb = -M, ub = M):
        print("adding continuous variable: var: {}, t: {}, lb: {}, ub: {}".format(var_name, time, lb, ub))
        self.variable.extend([{} for i in range(time + 1 - len(self.variable))])

        if self.solver == "Gurobi":
            self.variable[time][var_name] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name="{}_{}".format(var_name, time))
            self.model.update()
        elif self.solver == "Cplex":
            self.variable[time][var_name] = self.model.continuous_var(lb=lb, ub= ub, name="{}_{}".format(var_name, time))
        else: assert(False)
    def add_constraint(self, var, value):
        print("adding constraint: var: {}, value: {}".format(var, value))
        if self.mode == 'Boolean':
            if isinstance(value, (int, float)):
                if value <= 0:
                    if self.solver == "Gurobi":
                        self.model.addConstr(var == 1)
                    elif self.solver == "Cplex":
                        self.model.add_constraint(var == 1)
                    else: assert(False)
                else:
                    if self.solver == "Gurobi":
                        self.model.addConstr(var == 0)
                    elif self.solver == "Cplex":
                        self.model.add_constraint(var == 0)
                    else: assert(False)
            else:
                if self.solver == "Gurobi":
                    self.model.addConstr((var == 1) >> (value <= 0))
                    self.model.addConstr((var == 0) >> (value >= EPS))
                elif self.solver == "Cplex":
                    self.model.add_constraint((var == 1) >> (value <= 0))
                    self.model.add_constraint((var == 0) >> (value >= EPS))
                else: assert(False)
        elif self.mode == 'Quantitative':
            if self.solver == "Gurobi":
                self.model.addConstr(var == value)
            elif self.solver == "Cplex":
                self.model.add_constraint(var == value)
            else: assert(False)
        else: assert(False)
    def add_constraint_and(self, var, var_list):
        print("adding and constraint: var: {}, var_list: {}".format(var, var_list))
        if self.mode == 'Boolean':
            if self.solver == "Gurobi":
                self.model.addConstr(len(var_list) * var <= gp.quicksum(var_list))
                self.model.addConstr(len(var_list) - 1 + var >= gp.quicksum(var_list))
            elif self.solver == "Cplex":
                self.model.add_constraint(len(var_list) * var <= self.model.sum(var_list))
                self.model.add_constraint(len(var_list) - 1 + var >= self.model.sum(var_list))
            else: assert(False)
        elif self.mode == 'Quantitative':
            if self.solver == "Gurobi":
                self.model.addConstr(var == gp.min_(var_list))
            elif self.solver == "Cplex":
                self.model.add_constraint(var == 
            else: assert(False)
        else: assert(False)
    def add_constraint_or(self, var, var_list):
        print("adding or constraint: var: {}, var_list: {}".format(var, var_list))
        if self.mode == 'Boolean':
            if self.solver == "Gurobi":
                self.model.addConstr(var <= gp.quicksum(var_list))
                self.model.addConstr(len(var_list) * var >= gp.quicksum(var_list))
            elif self.solver == "Cplex":
                self.model.add_constraint(var <= self.model.sum(var_list))
                self.model.add_constraint(len(var_list) * var >= self.model.sum(var_list))
            else: assert(False)
        elif self.mode == 'Quantitative':
            if self.solver == "Gurobi":
                self.model.addConstr(var == gp.max_(var_list))
            elif self.solver == "Cplex":
            else: assert(False)
        else: assert(False)
    def add_dyn_constraint(self):
        """
        Adds constraints for the system dynamics given as a discrete-time state-space model with process and measurement noise to the main convex solver.
        """
        # Add convex variables
        for t in range(self.start_time, self.end_time+1):
            for i in range(self.dynamics.x_len):
                var_name = 'x[{}]_{}'.format(i, t)
                lb = self.dynamics.x_bounds[i,0]
                ub = self.dynamics.x_bounds[i,1]
                self.MILP_convex_var[var_name] = self.model.addVar(vtype=GRB.CONTINUOUS, lb = lb, ub = ub, name = var_name)
            
        for t in range(self.start_time, self.end_time+1):
            if self.dynamics.y_len is not None:
                for i in range(self.dynamics.y_len):
                    var_name = 'y[{}]_{}'.format(i, t)
                    lb = self.dynamics.y_bounds[i,0]
                    ub = self.dynamics.y_bounds[i,1]
                    self.MILP_convex_var[var_name] = self.model.addVar(vtype=GRB.CONTINUOUS, lb = lb, ub = ub, name = var_name)
        
        for t in range(self.start_time, self.end_time):
            if self.dynamics.u_len is not None:
                for i in range(self.dynamics.u_len):
                    var_name = 'u[{}]_{}'.format(i, t)
                    lb = self.dynamics.u_bounds[i,0]
                    ub = self.dynamics.u_bounds[i,1]
                    self.MILP_convex_var[var_name] = self.model.addVar(vtype=GRB.CONTINUOUS, lb = lb, ub = ub, name = var_name)

        # Update MILP convex solver
        self.model.update()

        # Add the initial states constraints to convex solver
        for i in range(self.dynamics.x_len):
            self.model.addConstr(self.MILP_convex_var['x[' + str(i) + ']_0'] == self.dynamics.x0[i, 0])

        # Update MILP convex solvers
        self.model.update()

        # Add convex constraints
        for t in range(self.start_time, self.end_time):
            self.model.addConstrs(self.MILP_convex_var['x[' + str(i) + ']_' + str(t+1)] \
                                                == gp.quicksum(self.dynamics.A[i,j]*self.MILP_convex_var['x[' + str(j) + ']_' + str(t)] for j in range(self.dynamics.x_len)) \
                                                    + gp.quicksum(self.dynamics.B[i,l]*self.MILP_convex_var['u[' + str(l) + ']_' + str(t)] for l in range(self.dynamics.u_len)) for i in range(self.dynamics.x_len))
            
            try:
                self.model.addConstrs(self.MILP_convex_var['y[' + str(i) + ']_' + str(t+1)] \
                                                    == gp.quicksum(self.dynamics.C[i,j]*self.MILP_convex_var['x[' + str(j) + ']_' + str(t)] for j in range(self.dynamics.x_len)) \
                                                        + gp.quicksum(self.dynamics.D[i,l]*self.MILP_convex_var['u[' + str(l) + ']_' + str(t)] for l in range(self.dynamics.u_len)) for i in range(self.dynamics.y_len))
            except:
                pass

        # Update MILP convex solver 
        self.model.update()

    # def add_NN_constraints(self):
    #   """
    #   Adds constraints for the NN to the main convex solver.
    #   """


