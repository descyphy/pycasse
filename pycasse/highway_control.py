import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from itertools import combinations

from pycasse.core import MILPSolver
from pycasse.contracts import contract
from pycasse.parser import Parser

M = 10**4
EPS = 10**-4
parser = Parser()

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

class Controller:
    #  PERCEPTION_RANGE = 15 # for intersection
    PERCEPTION_RANGE = 6.5 # for merging
    __slots__ = ('env', 'H', 'debug', 'model_dict', 'state_var_name', 'control_var_name', 'A', 'B', 'model', 'contract', 'control')

    def __init__(self, env, H, debug = False, plot = False):
        self.env = env
        self.H = H
        self.debug = debug

        self.state_var_name =  [ "d_x", "d_y", "v_x", "v_y" ]
        self.control_var_name = [ "a_x", "a_y" ]

        sim_freq = env.config["simulation_frequency"]
        self.A = [[1, 0, 1 / sim_freq, 0], [0, 1, 0, 1 / sim_freq], [0, 0, 1, 0], [0, 0, 0, 1]]
        self.B = [[0, 0], [0, 0], [1 / sim_freq, 0], [0, 1 / sim_freq]]

        self._construct_model()
        if plot:
            self.plot_path_range()

        self.control = [[] for v in self.env.controlled_vehicle]

    def _construct_model(self):
        """ Loads the setup of the highway environment.

        :param data: [description]
        :type data: [type]
        """

        self.model = MILPSolver()
        self.contract = contract("vehicle")

        velocity_bound = self.env.config["road"]["speed_limit"]
        acceleration_bound = self.env.config["road"]["acceleration_limit"]
        deter_vars = []
        deter_bounds = np.empty((0,2))

        for vehicle_id in range(len(self.env.controlled_vehicle)):
            # Find a list of deter variables
            x = [s + "_{}".format(vehicle_id) for s in self.state_var_name]
            u = [s + "_{}".format(vehicle_id) for s in self.control_var_name]
            deter_vars = deter_vars + x + u

            # Find the corresponding bounds for of deter variables
            if self.env.unwrapped.spec.id in ('highway-v1', 'merge-v1'):
                deter_bounds = np.append(deter_bounds, np.array([[-500, 500], [-500, 500], [0, velocity_bound], [-0.1*velocity_bound, 0.1*velocity_bound]]), axis=0)
                deter_bounds = np.append(deter_bounds, np.array([[-acceleration_bound, acceleration_bound], [-0.1*acceleration_bound, 0.1*acceleration_bound]]), axis=0)
            elif self.env.unwrapped.spec.id == 'intersection-v1':
                deter_bounds = np.append(deter_bounds, np.array([[-100, 100], [-100, 100], [-velocity_bound, velocity_bound], [-velocity_bound, velocity_bound]]), axis=0)
                deter_bounds = np.append(deter_bounds, np.array([[-acceleration_bound, acceleration_bound], [-acceleration_bound, acceleration_bound]]), axis=0)
            else: assert(False)
            

        self.contract.add_deter_vars(deter_vars, bounds = deter_bounds)
        # self.contract.printInfo()

    def optimize_model(self, current_states, group_num = None):
        """ Loads the setup of the highway environment.

        :param data: [description]
        :type data: [type]
        """

        # Add a model for each group of cooperating vehicles (or a vehicle)
        group_split = self._find_group(current_states, group_num)
        # print("group: {}".format(group_split))

        status = True
        for group in group_split:
            #  print(self.contract)
            # Find assumptions
            noncooperating_vehicle_num = len(self.env.controlled_vehicle) - len(group)
            
            if noncooperating_vehicle_num == 0:
                assumptions_formula = 'True'
            else:
                assumptions_condition = []
                for vehicle_id in range(len(self.env.controlled_vehicle)):
                    if vehicle_id not in group:
                        assumptions_condition.extend(["(({}_{} >= -0.1) & ({}_{} <= 0.1))".format(n, vehicle_id, n, vehicle_id) for n in self.control_var_name])
                        # assumptions_condition.extend(["({}_{} == 0)".format(n, vehicle_id) for n in self.control_var_name])
                if len(assumptions_condition) == 1:
                    assumptions_formula = "G[0,{}] {}".format(self.H, " & ".join(assumptions_condition))
                else:
                    assumptions_formula = "G[0,{}] ({})".format(self.H, " & ".join(assumptions_condition))

            # Find guarantees
            # Find no collision guarantees
            if len(self.env.controlled_vehicle) <= 1:
                guarantees_formula = 'True'
            else:
                vehicle_len = self.env.controlled_vehicle[0].DEFAULT_LENGTH
                vehicle_wid = self.env.controlled_vehicle[0].DEFAULT_WIDTH
                distance = np.linalg.norm(np.array([self.env.controlled_vehicle[0].DEFAULT_LENGTH, self.env.controlled_vehicle[0].DEFAULT_WIDTH]))
                # distance = max(self.env.controlled_vehicle[0].DEFAULT_LENGTH, self.env.controlled_vehicle[0].DEFAULT_WIDTH)
                
                #  print(distance)
                #  input()

                guarantees_condition = []
                # No collision between vehicles
                for (vehicle_num_1, vehicle_num_2) in combinations(range(len(self.env.controlled_vehicle)), 2):
                    if (vehicle_num_1 in group) or (vehicle_num_2 in group):
                            condition = []
                            if self.env.unwrapped.spec.id in ('highway-v1', 'merge-v1'):
                                condition.append("({}_{} - {}_{} >= {})".format(self.state_var_name[0], vehicle_num_1, self.state_var_name[0], vehicle_num_2, 2*vehicle_len))
                                condition.append("({}_{} - {}_{} >= {})".format(self.state_var_name[0], vehicle_num_2, self.state_var_name[0], vehicle_num_1, 2*vehicle_len))
                                condition.append("({}_{} - {}_{} >= {})".format(self.state_var_name[1], vehicle_num_1, self.state_var_name[1], vehicle_num_2, vehicle_wid))
                                condition.append("({}_{} - {}_{} >= {})".format(self.state_var_name[1], vehicle_num_2, self.state_var_name[1], vehicle_num_1, vehicle_wid))
                            elif self.env.unwrapped.spec.id == 'intersection-v1':
                                condition.append("({}_{} - {}_{} >= {})".format(self.state_var_name[0], vehicle_num_1, self.state_var_name[0], vehicle_num_2, distance))
                                condition.append("({}_{} - {}_{} >= {})".format(self.state_var_name[0], vehicle_num_2, self.state_var_name[0], vehicle_num_1, distance))
                                condition.append("({}_{} - {}_{} >= {})".format(self.state_var_name[1], vehicle_num_1, self.state_var_name[1], vehicle_num_2, distance))
                                condition.append("({}_{} - {}_{} >= {})".format(self.state_var_name[1], vehicle_num_2, self.state_var_name[1], vehicle_num_1, distance))
                                # condition.append("({}_{} - {}_{} >= {} + {}_{} - {}_{})".format(self.state_var_name[0], vehicle_num_1, self.state_var_name[0], vehicle_num_2, distance, self.state_var_name[2], vehicle_num_2, self.state_var_name[2], vehicle_num_1))
                                # condition.append("({}_{} - {}_{} >= {} + {}_{} - {}_{})".format(self.state_var_name[0], vehicle_num_2, self.state_var_name[0], vehicle_num_1, distance, self.state_var_name[2], vehicle_num_1, self.state_var_name[2], vehicle_num_2))
                                # condition.append("({}_{} - {}_{} >= {} + {}_{} - {}_{})".format(self.state_var_name[1], vehicle_num_1, self.state_var_name[1], vehicle_num_2, distance, self.state_var_name[3], vehicle_num_2, self.state_var_name[2], vehicle_num_1))
                                # condition.append("({}_{} - {}_{} >= {} + {}_{} - {}_{})".format(self.state_var_name[1], vehicle_num_2, self.state_var_name[1], vehicle_num_1, distance, self.state_var_name[3], vehicle_num_1, self.state_var_name[2], vehicle_num_2))
                            guarantees_condition.append("({})".format(" | ".join(condition)))
                if len(guarantees_condition) == 1:
                    guarantees_formula = "G[0,{}] {}".format(self.H, " & ".join(guarantees_condition))
                else:
                    guarantees_formula = "G[0,{}] ({})".format(self.H, " & ".join(guarantees_condition))

            # Set the contracts assumptions and guarantees
            self.contract.set_assume(assumptions_formula)
            self.contract.set_guaran(guarantees_formula)
            # self.contract.printInfo()
            
            # Set Dynamics
            self.model.add_contract(self.contract)
            for vehicle_id in range(len(self.env.controlled_vehicle)):
                # Find the state and control variables
                x = [s + "_{}".format(vehicle_id) for s in self.state_var_name]
                u = [s + "_{}".format(vehicle_id) for s in self.control_var_name]
                
                # Add dynamics
                self.model.add_dynamics(x = x, u = u, A = self.A, B = self.B)

            # Add the contract specifications
            self.model.add_constraint(parser(assumptions_formula)[0][0], name='b_a')
            self.model.add_constraint(parser(guarantees_formula)[0][0], name='b_g')

            # Region constraints
            for vehicle_id in group:
                vehicle = self.env.controlled_vehicle[vehicle_id]
                region_param = vehicle.region()
                region_param[np.abs(region_param) < EPS] = 0

                ego_x_var_name = "{}_{}".format(self.state_var_name[0], vehicle_id)
                ego_y_var_name = "{}_{}".format(self.state_var_name[1], vehicle_id)

                region_condition = []
                for region in region_param:
                    condition = []
                    for i in range(4):
                        condition.append("({}*{}^2 + {}*{} + {}*{}^2 + {}*{} + {} <= 0)".format(region[i][0], ego_x_var_name, region[i][1], ego_x_var_name, region[i][2], ego_y_var_name, region[i][3], ego_y_var_name, region[i][4]))
                    region_condition.append("({})".format(" & ".join(condition)))

                region_formula = "G[0,{}] ({})".format(self.H, " | ".join(region_condition))
                self.model.add_constraint(parser(region_formula)[0][0], name='b_r')
                # print(region_formula)

            # Add objective to MILP sovler
            objective_func = gp.LinExpr()

            # Fuel objectives (Sum of absolute values of u)
            acceleration_bound = self.env.config["road"]["acceleration_limit"]
            for vehicle_num in group:
                # The last step of acceleration is not related to the position or velocity at all
                # so we collect the acceleration without the last step
                for t in range(self.H):
                    # Find the gurobi variable
                    ax_var_name = self.model.deter_vars.index("a_x_{}__{}".format(vehicle_num, t))
                    ay_var_name = self.model.deter_vars.index("a_y_{}__{}".format(vehicle_num, t))
                    ax_var = expr2gurobiexpr(self.model, self.model.deter_vars_expr[ax_var_name])
                    ay_var = expr2gurobiexpr(self.model, self.model.deter_vars_expr[ay_var_name])

                    # Add fuel objectives in x and y axis
                    fuel_x = self.model.model.addVar(name="fuel_x_{}__{}".format(vehicle_num, t), lb = 0, ub=acceleration_bound)
                    fuel_y = self.model.model.addVar(name="fuel_y_{}__{}".format(vehicle_num, t), lb = 0, ub=acceleration_bound)
                    self.model.model.addConstr(ax_var <= fuel_x)
                    self.model.model.addConstr(-ax_var <= fuel_x)
                    self.model.model.addConstr(ay_var <= fuel_y)
                    self.model.model.addConstr(-ay_var <= fuel_y)
                    self.model.model.update()

                    # Add fuel objective in x and y axis
                    # objective_func += fuel_x + fuel_y
                    objective_func += (1+t)*fuel_x + (1+t)*fuel_y

            # Goal objectives
            for vehicle_num in group:
                for t in range(self.H + 1):
                    # Find the gurobi variable
                    dx_var_name = self.model.deter_vars.index("d_x_{}__{}".format(vehicle_num, t))
                    dy_var_name = self.model.deter_vars.index("d_y_{}__{}".format(vehicle_num, t))
                    dx_var = expr2gurobiexpr(self.model, self.model.deter_vars_expr[dx_var_name])
                    dy_var = expr2gurobiexpr(self.model, self.model.deter_vars_expr[dy_var_name])

                    # Add goal objectives in x and y axis
                    goal_x = self.model.model.addVar(name="goal_x_{}__{}".format(vehicle_num, t), lb = -M, ub = M)
                    goal_y = self.model.model.addVar(name="goal_y_{}__{}".format(vehicle_num, t), lb = -M, ub = M)
                    self.model.model.addConstr(dx_var - self.env.controlled_vehicle[vehicle_num].target[0] <= goal_x)
                    self.model.model.addConstr(self.env.controlled_vehicle[vehicle_num].target[0] - dx_var <= goal_x)
                    self.model.model.addConstr(dy_var - self.env.controlled_vehicle[vehicle_num].target[1] <= goal_y)
                    self.model.model.addConstr(self.env.controlled_vehicle[vehicle_num].target[1] - dy_var <= goal_y)
                    self.model.model.update()

                    # Add goal objective in x and y axis
                    if self.env.unwrapped.spec.id in ('highway-v1', 'merge-v1'):
                        objective_func += 2 * (len(self.env.controlled_vehicle)-vehicle_num) * goal_x + 200 * (len(self.env.controlled_vehicle)-vehicle_num) * goal_y
                    elif self.env.unwrapped.spec.id == 'intersection-v1':
                        if abs(current_states[vehicle_num][2]) >= 5*abs(current_states[vehicle_num][3]):
                            objective_func += 2 * (len(self.env.controlled_vehicle)-vehicle_num) * goal_x + 20 * (len(self.env.controlled_vehicle)-vehicle_num) * goal_y
                        elif 5*abs(current_states[vehicle_num][2]) <= abs(current_states[vehicle_num][3]):
                            objective_func += 20 * (len(self.env.controlled_vehicle)-vehicle_num) * goal_x + 2 * (len(self.env.controlled_vehicle)-vehicle_num) * goal_y
                        else:
                            objective_func += 2 * (len(self.env.controlled_vehicle)-vehicle_num)*goal_x + 2 * (len(self.env.controlled_vehicle)-vehicle_num) * goal_y

            # Set objectives
            # sl,elf.model.model.setObjective(objective_func, GRB.MAXIMIZE)
            self.model.model.setObjective(objective_func, GRB.MINIMIZE)

            # print(objective_func)

            for vehicle_num in range(len(self.env.controlled_vehicle)):
                # Get initial state variables
                d_x = self.model.model.getVarByName("d_x_{}__0".format(vehicle_num))
                d_y = self.model.model.getVarByName("d_y_{}__0".format(vehicle_num))
                v_x = self.model.model.getVarByName("v_x_{}__0".format(vehicle_num))
                v_y = self.model.model.getVarByName("v_y_{}__0".format(vehicle_num))

                # Add initial state constraints
                self.model.model.addConstr(d_x == current_states[vehicle_num][0], 'init_d_x_{}'.format(vehicle_num))
                self.model.model.addConstr(d_y == current_states[vehicle_num][1], 'init_d_y_{}'.format(vehicle_num))
                self.model.model.addConstr(v_x == current_states[vehicle_num][2], 'init_v_x_{}'.format(vehicle_num))
                self.model.model.addConstr(v_y == current_states[vehicle_num][3], 'init_v_y_{}'.format(vehicle_num))
            self.model.model.update()

            # Solve and fetch the solution for the control
            self.model.model.write("MILP_adaptive.lp")
            solved = self.model.solve()
            if self.debug and solved:
                self.model.print_solution()
            if not solved:
                self.model.model.computeIIS()
                self.model.model.write("MILP_adaptive.ilp")

            # Fetch the control
            if solved:
                for vehicle_num in group:
                    variables = [["a_x_{}__{}".format(vehicle_num, t), "a_y_{}__{}".format(vehicle_num, t)] for t in range(self.H)]
                    tmp_output = self.model.fetch_solution(variables, self.H)
                    self.control[vehicle_num] = tmp_output
            else:
                print("Failing vehicles: {}".format(group))
                self.model.model.computeIIS()
                self.model.model.write("MILP_adaptive.ilp")
                for vehicle_num in group:
                    self.control[vehicle_num] = [[0.0, 0.0] for t in range(self.H)]
            status &= solved
        
            self.model.reset()
            # input()

        return not status

    def _find_group(self, current_states, group_num):
        if group_num is not None:
            res = np.array_split(range(len(self.env.controlled_vehicle)), group_num)
        else:
            from pycasse.disjoint_set import DisjointSet
            ds = DisjointSet(list(range(len(self.env.controlled_vehicle))))
            #  print(current_states)
            for i in range(1, len(self.env.controlled_vehicle)):
                diff = current_states[:i, :2] - current_states[i, :2]
                diff = np.linalg.norm(diff, axis = 1)
                for j in range(len(diff)):
                    if diff[j] <= self.PERCEPTION_RANGE:
                        ds.union(i, j)
                        #  print("joining {}, {}".format(i, j))
            res = ds.result()

            if len(res) < len(self.env.controlled_vehicle):
                res = [list(range(len(self.env.controlled_vehicle)))]

        #  print(res)
        #  input()
        return res

    def find_control(self, time):
        """ Find control for all the groups of cooperating vehicles (or a vehicle if only one in the group).

        :param init: [description]
        :type init: [type]
        """
        control = [self.control[vehicle_id][time] for vehicle_id in range(len(self.env.controlled_vehicle))]

        return control
    
    def plot_path_range(self):
        """ Plots the pre-determined paths of the vehicles.
        """
        # Initialize plots and settings
        if len(self.env.controlled_vehicle) == 1:
            ax = [plt.gca()]
        elif self.env.unwrapped.spec.id in ('highway-v1', 'merge-v1'):
            _, ax = plt.subplots(len(self.env.controlled_vehicle), 1)
        elif self.env.unwrapped.spec.id == 'intersection-v1':
            _, ax = plt.subplots(1, len(self.env.controlled_vehicle))
        else: assert(False)

        cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

        # For each region of a vehicle, find vertices and plot the region
        for vehicle_id, vehicle in enumerate(self.env.controlled_vehicle):
            
            im = None
            region_param = vehicle.region()
            for region in region_param:
                f1 = lambda x,y : region[0,0]*x**2 + region[0,1]*x + region[0,2]*y**2 + region[0,3]*y + region[0,4]
                f2 = lambda x,y : region[1,0]*x**2 + region[1,1]*x + region[1,2]*y**2 + region[1,3]*y + region[1,4]
                f3 = lambda x,y : region[2,0]*x**2 + region[2,1]*x + region[2,2]*y**2 + region[2,3]*y + region[2,4]
                f4 = lambda x,y : region[3,0]*x**2 + region[3,1]*x + region[3,2]*y**2 + region[3,3]*y + region[3,4]

                if self.env.unwrapped.spec.id == 'highway-v1':
                    x = np.linspace(-10,200,1000)
                    y = np.linspace(-5,20,1000)
                elif self.env.unwrapped.spec.id == 'merge-v1':
                    x = np.linspace(0,250,1000)
                    y = np.linspace(-20,12,1000)
                elif self.env.unwrapped.spec.id == 'intersection-v1':
                    x = np.linspace(-50,50,1000)
                    y = np.linspace(-50,50,1000)
                else: assert(False)

                x,y = np.meshgrid(x,y)

                if im is None:
                    im = ((f1(x,y)<=0) & (f2(x,y)<=0) & (f3(x,y)<=0) & (f4(x,y)<=0)).astype(int) 
                else:
                    im += ((f1(x,y)<=0) & (f2(x,y)<=0) & (f3(x,y)<=0) & (f4(x,y)<=0)).astype(int) 

                ax[vehicle_id].set_title("Vehicle {}".format(vehicle_id), fontsize=10)

            #  print(im.nonzero())
            #  print(im.shape)
            ax[vehicle_id].imshow(im, extent=(x.min(),x.max(),y.min(),y.max()), origin="lower", cmap=cmaps[vehicle_id])
                
        plt.savefig("{}_path.png".format(self.env.unwrapped.spec.id))
