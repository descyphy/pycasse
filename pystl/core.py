from copy import deepcopy
import cplex
from gurobipy import GRB
import gurobipy as gp
import numpy as np
from scipy.stats import norm
from z3 import *

from pystl.contracts import *
from pystl.parser import *

class Preprocess:
    def __init__(self, solver, debug = False):
        self.idx = 0
        self.solver = solver
        self.debug = debug

    def __call__(self):
        end_time = 0
        # print(self.solver.constraints)
        # input()
        for i, c in enumerate(self.solver.constraints):
            (self.solver.constraints[i], t) = self.preprocess(c, 1)
            end_time = max(end_time, t)
        return self.idx, end_time

    def preprocess(self, node, end_time):
        if self.debug:
            print(repr(node))
            print(end_time)
        if (node.ast_type == "Not"):
            #  print("remove negation")
            res = node.formula_list[0]

            res.invert_ast_type()

            if res.ast_type == "AP":
                res.expr = -1 * res.expr + EPS
                assert(len(res.formula_list) == 0)
            elif res.ast_type == "StAP":
                res.expr = -1 * res.expr + EPS
                res.prob = 1 - res.prob
                assert(len(res.formula_list) == 0)
            else:
                for i, f in enumerate(res.formula_list):
                    res.formula_list[i] = ~f
        else:
            #  print("ignore")
            res = node

        res.idx = self.idx
        self.idx += 1

        if self.debug:
            print("start traverse")
            print(repr(res))
            input()
        if res.ast_type in ('G', 'F'):
            (res.formula_list[0], end_time) = self.preprocess(res.formula_list[0], end_time + res.interval[1])
        elif res.ast_type in ('U', 'R'):
            (res.formula_list[0], e1) = self.preprocess(res.formula_list[0], end_time + res.interval[1])
            (res.formula_list[1], e2) = self.preprocess(res.formula_list[1], end_time + res.interval[1])
            end_time = max(e1, e2)
        elif res.ast_type in ('And', 'Or'):
            current_end_time = end_time
            for i, f in enumerate(res.formula_list):
                (res.formula_list[i], e) = self.preprocess(f, current_end_time)
                end_time = max(e, end_time)

        if self.debug:
            print(repr(res))
            print(end_time)
            print("end traverse")
            input()
        return (res, end_time)

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
        self.switching_dynamics = None
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

    def add_switching_dynamics(self, dynamics):
        """
        Adds a dynamics to the SMC solver.
        """
        self.switching_dynamics = dynamics

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
    __slots__ = ('verbose', 'mode', 'solver', 'objective', 'debug', 'contract', 'hard_constraints', 'soft_constraints', 'soft_constraints_info', 'soft_constraints_var', 'dynamics', 'switching_dynamics', 'switching_time',
            'solver', 'model', 'idx', 'node_variable', 'contract_variable')

    def __init__(self, verbose=False, mode = "Boolean", solver = "Gurobi", debug = False):
        assert(mode in ("Boolean", "Quantitative"))
        assert(solver in ("Gurobi", "Cplex"))
        #  print("====================================================================================")
        #  print("Initializing MILP solver...")

        self.verbose   = verbose
        self.mode      = mode
        self.solver    = solver
        self.objective = []
        self.debug     = debug
        self.reset()

    def reset(self):
        # Initialize attributes
        self.contract    = None
        self.hard_constraints = []
        self.soft_constraints = []
        self.soft_constraints_info = []
        self.soft_constraints_var = []
        self.dynamics    = []
        self.switching_dynamics = []
        self.switching_time = None

        # Initialize convex solvers
        if self.solver == "Gurobi":
            self.model = gp.Model() # Convex solver for solving the MILP convex problem
            # self.model.setParam("NonConvex", 2)
            if not self.verbose:
                self.model.setParam("OutputFlag", 0)
        elif self.solver == "Cplex":
            self.model = cplex.Cplex()
        else: assert(False)

        # Initialize dictionary of Gurobi variables
        self.node_variable     = np.empty(0)
        self.contract_variable = np.empty(0)

    @property
    def constraints(self):
        return self.hard_constraints + self.soft_constraints

    def add_contract(self, contract):
        """
        Adds a contract to the SMC solver.
        """
        self.contract = contract

    def add_hard_constraint(self, constraint):
        """
        Adds a hard constrint to the MILP solver.
        """
        if (isinstance(constraint, str)): # If the constraint is given as a string
            parser = Parser(self.contract) # Parse the string into an AST
            constraint = parser(constraint)
        elif (isinstance(constraint, ASTObject)): # If the constraint is given as an AST
            constraint = constraint

        self.hard_constraints.append(constraint)

    def add_soft_constraint(self, constraint, region_num=None, time=None):
        """
        Adds a soft constrint to the MILP solver.
        """
        if (isinstance(constraint, str)): # If the constraint is given as a string
            parser = Parser(self.contract) # Parse the string into an AST
            constraint = parser(constraint)
        elif (isinstance(constraint, ASTObject)): # If the constraint is given as an AST
            constraint = constraint

        self.soft_constraints.append(constraint)
        self.soft_constraints_info.append([region_num, time])

    def add_dynamic(self, dynamic):
        """
        Adds a dynamics to the SMC solver.
        """
        self.dynamics.append(dynamic)

    def add_switching_dynamic(self, switching_dynamic, switching_time=None):
        """
        Adds a dynamics to the SMC solver.
        """
        self.switching_dynamics = switching_dynamic
        if switching_time is not None:
            self.switching_time = switching_time

    def preprocess(self):
        (num_node, end_time) = Preprocess(self)()
        if self.solver == "Gurobi":
            self.node_variable = -1 * np.ones((num_node, end_time), dtype = object)
            self.contract_variable = -1 * np.ones((len(self.contract.deter_var_list)+1, end_time), dtype = object) # -1 is to exclude constant variable
        elif self.solver == "Cplex":
            self.node_variable = -1 * np.ones((num_node, end_time), dtype = object)
            self.contract_variable = -1 * np.ones((len(self.contract.deter_var_list)+1, end_time), dtype = object)
        else: assert(False)

        if self.debug:
            for c in self.hard_constraints:
                print(repr(c))
            print("node variable shape: {}".format(self.node_variable.shape))
            print("contract variable shape: {}".format(self.contract_variable.shape))
            input()

        for c in self.hard_constraints:
            self.set_constraint(c, True)

        count = 0
        for c in self.soft_constraints:
            self.set_constraint(c, False, soft_idx=count)
            count += 1

        for d in self.dynamics:
            self.set_dynamic(d.vector)

        # self.set_switching_dynamic()

        return end_time

    def set_objective(self, objective):
        if self.solver == "Gurobi":
            self.model.setObjective(objective, GRB.MINIMIZE)
            self.model.update()
        #  elif self.solver == "Cplex":
        #      self.model.minimize(objective)
        else: assert(False)

    def solve(self):
        """ Solves the MILP problem """
        # Add constraints of the contract and dynamics to the SAT, main, and SSF convex solver
        assert(len(self.hard_constraints) > 0)

        # if sense == 'min':
        #     if self.solver == "Gurobi":
        #         self.model.setObjective(self.variable[0][parse_tree_root.name], GRB.MINIMIZE)
        #     # elif self.solver == "Cplex":
        #     #     self.model.minimize(self.variable[0][parse_tree_root.name])
        #     else: assert(False)
        # elif sense == 'max':
        #     if self.solver == "Gurobi":
        #         self.model.setObjective(self.variable[0][parse_tree_root.name], GRB.MAXIMIZE)
        #     # elif self.solver == "Cplex":
        #     #     self.model.maximize(self.variable[0][parse_tree_root.name])
        #     else: assert(False)

        # Solve the optimization problem
        if self.solver == "Gurobi":
            self.model.write('MILP.lp')
            self.model.optimize()
        elif self.solver == "Cplex":
            self.model.write('MILP.lp')
            self.model.solve()
        else: assert(False)

        # Print solution
        if (self.solver == 'Gurobi' and self.model.getAttr("Status") == 2) or (self.solver == 'Cplex' and self.model.solution.get_status() in (1,101)): # If MILP is successfully solved,
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

    def set_constraint(self, constraint, hard, soft_idx=None):
        """
        Adds contraints to the solvers.
        """
        # Build the parse tree
        self.set_node_constraint(constraint)

        if self.mode == 'Boolean':
            if self.solver == "Gurobi":
                if hard:
                    self.model.addConstr(self.node_variable[constraint.idx, 0] == 1)
                else:
                    self.soft_constraints_var.append([self.node_variable[constraint.idx, 0], self.soft_constraints_info[soft_idx][0], self.soft_constraints_info[soft_idx][1]])
                    
            #  elif self.solver == "Cplex":
            #      self.model.linear_constraints.add(lin_expr = [[[self.node_variable[constraint.idx, 0]], [1]]], senses = "E", rhs = [1])
            else: assert(False)
        elif self.mode == 'Quantitative':
            if self.solver == "Gurobi":
                if hard:
                    self.model.addConstr(self.node_variable[constraint.idx, 0] >= 0)
                else:
                    self.soft_constraints_var.append([self.node_variable[constraint.idx, 0], self.soft_constraints_info[soft_idx][0], self.soft_constraints_info[soft_idx][1]])

            #  elif self.solver == "Cplex":
            #      self.model.linear_constraints.add(lin_expr = [[[self.node_variable[constraint.idx, 0]], [1]]], senses = "G", rhs = [0])
            else: assert(False)

    def set_node_constraint(self, node, start_time = 0, end_time = 1):
        """
        Adds SAT and convex constraints from a contract to SAT and convex solvers.
        """
        if (self.debug):
            print("setting node: {}".format(repr(node)))
            input()

        # 1. Add Boolean variables to the MILP convex solver
        for t in range(start_time, end_time):
            if self.mode == 'Boolean':
                self.model_add_binary_variable(node.idx, t)
            elif self.mode == 'Quantitative':
                self.model_add_continous_variable(node.idx, t)
            else: assert(false)
        if (self.debug):
            print(self.node_variable)
            input()

        #  2. construct constraints according to the operation type
        if node.ast_type in ('AP', 'StAP'):
            self.set_ap_stap_constraint(node, start_time, end_time)
        elif node.ast_type in ('G', 'F'): # Temporal unary operators
            self.set_G_F_constraint(node, start_time, end_time)
        elif node.ast_type in ('U', 'R'): # Temporal binary operator
            self.set_U_R_constraint(node, start_time, end_time)
        elif node.ast_type in ('And', 'Or'):
            self.set_and_or_constraint(node, start_time, end_time)
        elif node.ast_type in ('True'):
            self.set_true_constraint(node, start_time, end_time)
        elif node.ast_type in ('False'):
            self.set_false_constraint(node, start_time, end_time)
        elif node.ast_type == 'Not': # Non-temporal unary operator
            assert(False)
        else: assert(False)

    def set_ap_stap_constraint(self, node, start_time = 0, end_time = 1):
        if (self.debug):
            print("set constraints for AP or StAP")

        #  1. construct the expression for constraint: expr <= 0
        expr = node.expr.deter_data.copy()
        constant = node.expr.constant
        if node.ast_type == 'StAP': # If the current node is a stochastic AP,
            assert(node.expr.nondeter_data.shape[1] == 1)
            data = node.expr.nondeter_data[:, 0]
            data_size = node.expr.nondeter_data.shape[0]
            expectation = data @ self.contract.nondeter_var_mean[:data_size]
            variance = data @ self.contract.nondeter_var_cov[:data_size, :data_size] @ data.T
            #  print(data)
            #  print(self.contract.nondeter_var_mean)
            #  print(self.contract.nondeter_var_cov)
            #  print(expectation)
            #  print(variance)

            constant += expectation + norm.ppf(node.probability()) * math.sqrt(variance)

        variables = np.nonzero(expr)

        if self.debug:
            print(expr)

        #  2. add variables of contract in every concerned time period to model
        for t in range(start_time, end_time):
            for v in variables[0]:
                if self.contract.deter_var_list[v-1].data_type == 'BINARY':
                    self.model_add_binary_variable(v, t, var_type = 'contract')
                elif self.contract.deter_var_list[v-1].data_type == 'CONTINUOUS':
                    lb = self.contract.deter_var_list[v-1].bound[0]
                    ub = self.contract.deter_var_list[v-1].bound[1]
                    self.model_add_continous_variable(v, t, var_type = 'contract', lb = lb, ub = ub)
                else: assert(False)

            if (self.debug):
                print(self.contract_variable)


            # 3. Add convex constraints
            self.model_add_inequality_constraint(node.idx, t, expr, constant)
            if (self.debug):
                input()

    def set_G_F_constraint(self, node, start_time = 0, end_time = 1):
        if (self.debug):
            print("set constraints for G or F")
        self.set_node_constraint(node.formula_list[0], start_time=start_time+node.interval[0], end_time=end_time+node.interval[1])
        
		# Build tmp_prop_formula to encode the logic
        for t in range(start_time, end_time):
            if node.ast_type == 'G': # Globally
                self.model_add_constraint_and(self.node_variable[node.idx, t], self.node_variable[node.formula_list[0].idx, t + node.interval[0]: t + node.interval[1] + 1].tolist())
            elif node.ast_type == 'F': # Finally
                self.model_add_constraint_or(self.node_variable[node.idx, t], self.node_variable[node.formula_list[0].idx, t + node.interval[0]: t + node.interval[1] + 1].tolist())
            else: assert(False)
        
        if (self.debug):
            print("afer setting constraints for And or Or")
            print(node)
            input()

    def set_U_R_constraint(self, node, start_time = 0, end_time = 1):
        if (self.debug):
            print("set constraints for U or R")
        self.set_node_constraint(node.formula_list[0], start_time=start_time, end_time=end_time+node.interval[1] - 1)
        self.set_node_constraint(node.formula_list[1], start_time=start_time+node.interval[0], end_time=end_time+node.interval[1])

        for t in range(start_time, end_time):
            aux_variables = []
            for t_i in range(t + node.interval[0], t + node.interval[1] + 1):
                if self.mode == 'Boolean':
                    aux_variable = self.model_add_binary_variable_by_name("node_{}_aux_{}_{}".format(node.idx, t, t_i))
                elif self.mode == 'Quantitative':
                    aux_variable = self.model_add_continuous_variable_by_name("node_{}_aux_{}_{}".format(node.idx, t, t_i))
                else: assert(False)

                if node.ast_type == 'U': # Globally
                    self.model_add_constraint_and(aux_variable, self.node_variable[node.formula_list[0].idx, t:t_i].tolist() + [self.node_variable[node.formula_list[1].idx, t_i]])
                elif node.ast_type == 'R': # Globally
                    self.model_add_constraint_or(aux_variable, self.node_variable[node.formula_list[0].idx, t:t_i].tolist() + [self.node_variable[node.formula_list[1].idx, t_i]])
                else: assert(False)

                aux_variables.append(aux_variable)

            if node.ast_type == 'U': # Globally
                # Add anxiliary Boolean constraints to MILP convex solver
                self.model_add_constraint_or(self.node_variable[node.idx, t], aux_variables)
            elif node.ast_type == 'R': # Globally
                self.model_add_constraint_and(self.node_variable[node.idx, t], aux_variables)
            else: assert(False)
            if self.debug:
                input()

    def set_and_or_constraint(self, node, start_time = 0, end_time = 1):
        # Recursions
        if (self.debug):
            print("set constraints for And or Or")
        for subformula in node.formula_list:
            self.set_node_constraint(subformula, start_time=start_time, end_time=end_time)

        # Add Boolean constraints to MILP convex solver
        for t in range(start_time, end_time):
            if node.ast_type == 'And': # AND
                self.model_add_constraint_and(self.node_variable[node.idx, t], [self.node_variable[f.idx,t] for f in node.formula_list])
            elif node.ast_type == 'Or': # OR
                self.model_add_constraint_or(self.node_variable[node.idx, t], [self.node_variable[f.idx,t] for f in node.formula_list])
            else: assert(False)
        if (self.debug):
            print("afer setting constraints for And or Or")
            print(node)
            input()

    def set_true_constraint(self, node, start_time = 0, end_time = 1):
        # Recursions
        if (self.debug):
            print("set constraints for true")

        for t in range(start_time, end_time):
            variable_idx = np.array([[node.idx], [t]])
            if self.mode == 'Boolean':
                self.model_add_constraint(variable_idx, np.ones(1), 1, 'node')
            elif self.mode == 'Quantitative':
                self.model_add_constraint(variable_idx, np.ones(1), M, 'node')
            else: assert(False)
        
        if (self.debug):
            print("afer setting constraints for true")
            print(node)
            input()

    def set_false_constraint(self, node, start_time = 0, end_time = 1):
        # Recursions
        if (self.debug):
            print("set constraints for false")

        for t in range(start_time, end_time):
            variable_idx = np.array([[node.idx], [t]])
            if self.mode == 'Boolean':
                self.model_add_constraint(variable_idx, np.ones(1), 0, 'node')
            elif self.mode == 'Quantitative':
                self.model_add_constraint(variable_idx, np.ones(1), -M, 'node')
            else: assert(False)
        
        if (self.debug):
            print("afer setting constraints for false")
            print(node)
            input()

    def set_dynamic(self, dynamic):
        if self.debug:
            print(dynamic)
            input()
        
        for row in range(len(dynamic.data)):
            variable = [[],[]]
            multiplier = []
            rhs = 0
            for key, value in dynamic.var2id.items():
                if dynamic.data[row, value] != 0:
                    if key[0] != 0:
                        variable[0].append(key[0])
                        variable[1].append(key[1])
                        multiplier.append(dynamic.data[row,value])
                    else:
                        assert(key[1] == 0)
                        rhs += dynamic.data[row,value]
            variable = np.array(variable)
            multiplier = np.array(multiplier)
            if self.debug:
                print("variable: {}".format(variable))
                print("multiplier: {}".format(multiplier))
                print("rhs: {}".format(rhs))

            for _ in range(self.contract_variable.shape[1] - dynamic.max_time):
                for i in range(variable.shape[1]):
                    v = variable[0,i]-1
                    t = variable[1,i]
                    if self.contract.deter_var_list[v].data_type == 'BINARY':
                        self.model_add_binary_variable(v, t, var_type = 'contract')
                    elif self.contract.deter_var_list[v].data_type == 'CONTINUOUS':
                        lb = self.contract.deter_var_list[v].bound[0]
                        ub = self.contract.deter_var_list[v].bound[1]
                        self.model_add_continous_variable(v+1, t, var_type = 'contract', lb = lb, ub = ub)
                    else: assert(False)
                self.model_add_constraint(variable, multiplier, -rhs)
                variable[1] +=1
        
        if (self.debug):
            input()

    def set_switching_dynamic(self):
        max_time = self.contract_variable.shape[1]
        
        bool_vars = self.model.addVars(max_time, vtype=GRB.BINARY, name='b_switching')
        self.model.update()

        # print(self.switching_dynamics)

        for k in [0, 1]:
            for dynamic in self.switching_dynamics[k]:
                tmp_dyn = dynamic.vector
                for row in range(len(tmp_dyn.data)):
                    variable = [[],[]]
                    multiplier = []
                    rhs = 0
                    for key, value in tmp_dyn.var2id.items():
                        if tmp_dyn.data[row, value] != 0:
                            if key[0] != 0:
                                variable[0].append(key[0])
                                variable[1].append(key[1])
                                multiplier.append(tmp_dyn.data[row,value])
                            else:
                                assert(key[1] == 0)
                                rhs += tmp_dyn.data[row,value]
                    variable = np.array(variable)
                    multiplier = np.array(multiplier)
                    if self.debug:
                        print("variable: {}".format(variable))
                        print("multiplier: {}".format(multiplier))
                        print("rhs: {}".format(rhs))

                    for j in range(self.contract_variable.shape[1] - tmp_dyn.max_time):
                        for i in range(variable.shape[1]):
                            v = variable[0,i]-1
                            t = variable[1,i]
                            #  print(v, t)
                            if self.contract.deter_var_list[v].data_type == 'BINARY':
                                self.model_add_binary_variable(v, t, var_type = 'contract')
                            elif self.contract.deter_var_list[v].data_type == 'CONTINUOUS':
                                lb = self.contract.deter_var_list[v].bound[0]
                                ub = self.contract.deter_var_list[v].bound[1]
                                self.model_add_continous_variable(v, t, var_type = 'contract', lb = lb, ub = ub)
                            else: assert(False)
                        #  print(self.contract_variable)
                        self.model.update()
                        if self.switching_time is not None:
                            if j < self.switching_time:
                                self.model_add_switching_constraint(bool_vars[j], k, variable, multiplier, -rhs, value=0)
                            else:
                                self.model_add_switching_constraint(bool_vars[j], k, variable, multiplier, -rhs, value=1)
                        else:
                            self.model_add_switching_constraint(bool_vars[j], k, variable, multiplier, -rhs)
                        variable[1] +=1
        
        if (self.debug):
            input()

    def model_add_binary_variable(self, idx, time, var_type = 'node'):
        assert(var_type in ('node', 'contract'))
        
        if (self.debug):
            print("adding binary variable: type: {}, idx: {}, t: {}".format(var_type, idx, time))

        if (var_type == 'node'):
            variable_array = self.node_variable
        elif (var_type == 'contract'):
            variable_array = self.contract_variable
        else: assert(False)

        if (not isinstance(variable_array[idx, time], gp.Var) and (variable_array[idx, time] == -1)):
            variable_array[idx, time] = self.model_add_binary_variable_by_name("{}_{}_{}".format(var_type, idx, time))

    def model_add_binary_variable_by_name(self, name):
        if self.solver == "Gurobi":
            res = self.model.addVar(vtype=GRB.BINARY, name=name)
            self.model.update()
            return res
        elif self.solver == "Cplex":
            (idx,) = (self.model.variables.add(names = [name], types = [self.model.variables.type.binary]))
            return idx
        else: assert(False)

    def model_add_continous_variable(self, idx, time, lb = -M, ub = M, var_type = 'node'):
        if (self.debug):
            print("adding continuous variable: type: {}, idx: {}, t: {}, lb:{}, ub:{}".format(var_type, idx, time, lb, ub))

        if (var_type == 'node'):
            variable_array = self.node_variable
        elif (var_type == 'contract'):
            variable_array = self.contract_variable
        else: assert(False)

        if (not isinstance(variable_array[idx, time], gp.Var) and (variable_array[idx, time] == -1)):
            variable_array[idx, time] = self.model_add_continuous_variable_by_name("{}_{}_{}".format(var_type, idx, time), lb, ub)

    def model_add_continuous_variable_by_name(self, name, lb = -M, ub = M):
        if self.solver == "Gurobi":
            res = self.model.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name=name)
            self.model.update()
            return res
        elif self.solver == "Cplex":
            (idx,) = self.model.variables.add(names = [name], types = [self.model.variables.type.continuous], lb=[lb], ub=[ub])
            return idx
        else: assert(False)

    def model_add_inequality_constraint(self, node_idx, time, expr, constant):
        if (self.debug):
            print("adding inequality constraint: node id: {}, time: {}, expr: {}, constant: {}".format(node_idx, time, expr, constant))
        constr = constant
        
        #  1. handle linear term
        mask = np.nonzero(expr[:,0])[0]
        if mask.size > 0:
            constr += np.sum(self.contract_variable[mask, time] * expr[mask, 0])

        #  2. handle quadratic term
        if expr.shape[1] >= 2:
            mask = np.nonzero(expr[:,1])[0]
            for m in mask:
                multiplier = expr[m, 1]
                # constr += multiplier * (self.contract_variable[m, time] ** 2)
                constr += multiplier * (self.contract_variable[m, time] * self.contract_variable[m, time])

        if self.debug:
            print("constraint: {}".format(constr <= 0))
            #  print("mask: {}".format(mask))
            #  print("variable: {}".format(variable))
            #  print("multiplier: {}".format(multiplier))
            #  print("rhs: {}".format(rhs))

        if self.mode == 'Boolean':
            if self.solver == "Gurobi":
                self.model.addConstr(constr <= M * (1 - self.node_variable[node_idx, time]))
                # self.model.addConstr(constr >= EPS - M * (self.node_variable[node_idx, time]))
                self.model.addConstr(constr >= - M * (self.node_variable[node_idx, time]))
            #  elif self.solver == "Cplex":
            #      lin_expr = [[[self.node_variable[node_idx, time]] + variable.tolist(), [M] + multiplier.tolist()]]
            #      self.model.linear_constraints.add(lin_expr = (lin_expr * 2), senses = "LG", rhs = [-rhs+M, -rhs+EPS])
            else: assert(False)
        elif self.mode == 'Quantitative':
            if self.solver == "Gurobi":
                self.model.addConstr(self.node_variable[node_idx, time] == constr)
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

    def model_add_switching_constraint(self, bool, bool_value, variable, multiplier, rhs, var_type = 'contract', value = None):
        if (var_type == 'node'):
            variable_array = self.node_variable
        elif (var_type == 'contract'):
            variable_array = self.contract_variable
        else: assert(False)

        if (self.debug):
            print("adding constraint: variable: {}, multiplier: {}, rhs: {}".format(variable_array[variable[0], variable[1]], multiplier, rhs))

        if self.solver == "Gurobi":
            self.model.addConstr((bool == bool_value) >> (np.sum(variable_array[variable[0], variable[1]] * multiplier) == rhs))
            if value is not None:
                self.model.addConstr(bool == value)
            self.model.update()

        elif self.solver == "Cplex":
            lin_expr = [[variable_array[variable[0], variable[1]].tolist(), multiplier.tolist()]]
            self.model.linear_constraints.add(lin_expr = lin_expr, senses = "E", rhs = [rhs])
        else: assert(False)

    def model_add_constraint_and(self, var, var_list):
        if (self.debug):
            print("adding and constraint: var: {}, expr: {}".format(var, var_list))

        if self.mode == 'Boolean':
            if self.solver == "Gurobi":
                self.model.addConstr(var == gp.and_(var_list))
            elif self.solver == "Cplex":
                var_size = len(var_list)
                lin_expr1 = [[[var] + var_list, [1] + [-1 if j == i else 0 for j in range(var_size)]] for i in range(var_size)]
                self.model.linear_constraints.add(lin_expr = lin_expr1, senses = "L" * var_size, rhs = [0] * var_size)
                lin_expr2 = [[[var] + var_list, [1] + [-1] * var_size]]
                self.model.linear_constraints.add(lin_expr = lin_expr2, senses = "G", rhs = [1 - var_size])
            else: assert(False)

        elif self.mode == 'Quantitative':
            if self.solver == "Gurobi":
                self.model.addConstr(var == gp.min_(var_list))
            elif self.solver == "Cplex":
                var_size = len(var_list)
                lin_expr1 = [[[var] + var_list, [1] + [-1 if j == i else 0 for j in range(var_size)]] for i in range(var_size)]
                self.model.linear_constraints.add(lin_expr = lin_expr1, senses = "L" * var_size, rhs = [0] * var_size)
            else: assert(False)

        else: assert(False)

    def model_add_constraint_or(self, var, var_list):
        if (self.debug):
            print("adding or constraint: var: {}, expr: {}".format(var, var_list))

        if self.mode == 'Boolean':
            if self.solver == "Gurobi":
                self.model.addConstr(var == gp.or_(var_list))
            elif self.solver == "Cplex":
                var_size = len(var_list)
                lin_expr1 = [[[var] + var_list, [-1] + [1 if j == i else 0 for j in range(var_size)]] for i in range(var_size)]
                self.model.linear_constraints.add(lin_expr = lin_expr1, senses = "L" * var_size, rhs = [0] * var_size)
                lin_expr2 = [[[var] + var_list, [1] + [-1] * var_size]]
                self.model.linear_constraints.add(lin_expr = lin_expr2, senses = "L", rhs = [0])
            else: assert(False)

        elif self.mode == 'Quantitative':
            if self.solver == "Gurobi":
                self.model.addConstr(var == gp.max_(var_list))
            elif self.solver == "Cplex":
                var_size = len(var_list)
                lin_expr1 = [[[var] + var_list, [-1] + [1 if j == i else 0 for j in range(var_size)]] for i in range(var_size)]
                self.model.linear_constraints.add(lin_expr = lin_expr1, senses = "L" * var_size, rhs = [0] * var_size)
            else: assert(False)

        else: assert(False)

    def print_solution(self):
        (len_var, len_t) = self.contract_variable.shape
        for v in range(1,len_var):
            for t in range(len_t):
                if (isinstance(self.contract_variable[v,t], gp.Var) or (self.contract_variable[v,t] != -1)):
                    if self.solver == "Gurobi":
                        print("{}_{}: {}".format(self.contract.deter_var_list[v-1].name, t, self.contract_variable[v,t].x))
                    elif self.solver == "Cplex":
                        print("{}_{}: {}".format(self.contract.deter_var_list[v-1].name, t, self.model.solution.get_values()[self.contract_variable[v,t]]))
                    else: assert(False)
        
        # for t in range(t):
        #     var = self.model.getVarByName("regions_{}".format(t))
        #     print("regions_{}: {}".format(t, var.x))
        #     var = self.model.getVarByName("goal_y_{}".format(t))
        #     print("goal_y_{}: {}".format(t, var.x))
        # input()
                    
        
        # for t in range(len_t):
        #     if self.solver == "Gurobi":
        #         print("b_switching[{}]: {}".format(t, self.model.getVarByName("b_switching[{}]".format(t)).x))

        #  (len_var, len_t) = self.node_variable.shape
        #  for v in range(len_var):
        #      for t in range(len_t):
        #          if (isinstance(self.node_variable[v,t], gp.Var) or (self.node_variable[v,t] != -1)):
        #              if self.solver == "Gurobi":
        #                  print("node_{}_{}: {}".format(v, t, self.node_variable[v,t].x))
        #              elif self.solver == "Cplex":
        #                  print("node_{}_{}: {}".format(v, t, self.model.solution.get_values()[v]))
        #                  print("node_{}_{}: {}".format(v, t, self.model.solution.get_values()[self.node_variable[v,t]]))
        #              else: assert(False)
        print()
    
    def fetch_control(self, controlled_vars):
        """ 
        Fetches the values of the controlled variables, if available.
        """
        # Initialize output
        output = []

        # Find the output
        (_, len_t) = self.contract_variable.shape
        for var in controlled_vars:
            tmp_output = []
            for t in range(len_t):
                if isinstance(self.contract_variable[var.idx,t], gp.Var):
                    if self.solver == "Gurobi":
                        # print("{}_{}".format(self.contract.deter_var_list[var.idx-1].name, t))
                        # print(self.contract_variable[var.idx, t].x)
                        tmp_output.append(self.contract_variable[var.idx, t].x)
            output.append(tmp_output)
                    
        return output

