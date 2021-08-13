import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
from itertools import combinations

from pystl.variable import M, EPS
from pystl.vector import Vector, Next
from pystl.core import MILPSolver
from pystl.contracts import contract

class highway_env_controller:
    __slots__ = ('environment', 'model_dict', 'vehicles', 'state_names', 'control_names', 'horizon')

    def __init__(self, highway_setup_data, horizon, debug = False):
        self.environment = ''
        self.model_dict = {}
        self.vehicles = []
        self.state_names = []
        self.control_names = []
        self.horizon = 0

        self.load_setup(highway_setup_data, horizon)

    def load_setup(self, data, horizon, savepath=True):
        """ Loads the setup of the highway environment.

        :param data: [description]
        :type data: [type]
        """
        
        # Fetch information on the environment
        self.environment = data["scenario"]
        self.vehicles = data["vehicle"]["id"]
        self.state_names = data["dynamics"]["x"]
        self.control_names = data["dynamics"]["u"]
        self.horizon = horizon

        # Find dynamics
        dt = float(data["dynamics"]["dt"])
        A = np.array(data["dynamics"]["A"])
        B = np.array(data["dynamics"]["B"])
        A = np.where(A=="dt", dt, A).astype(float)
        B = np.where(B=="dt", dt, B).astype(float)

        # Plot and save the pre-determined paths of the vehicles in the environment
        if savepath:
            self.plot_path(data["vehicle"]["region"]["equation"])

        # Add a model for each group of cooperating vehicles (or a vehicle)
        for group in data["vehicle"]["group"]:
            #  print(group)
            # Build a MILP solver
            tmp_model = MILPSolver()

            # Initialize a contract
            contract_name = "vehicle_{}".format("_".join([str(i) for i in group]))
            tmp_contract = contract(contract_name)

            # Set deterministic uncontrolled and controlled variables
            velocity_bound = data["physics"]["velocity_bound"]
            acceleration_bound = data["physics"]["acceleration_bound"]
            uncontrolled_vars = []
            controlled_vars = []
            uncontrolled_bounds = np.empty((0,2))
            controlled_bounds = np.empty((0,2))

            for tmp_vehicle_num in self.vehicles:
                x_tmp = [s + "_{}".format(tmp_vehicle_num) for s in self.state_names]
                u_tmp = [s + "_{}".format(tmp_vehicle_num) for s in self.control_names]
                uncontrolled_vars = uncontrolled_vars + x_tmp
                if self.environment in ("highway", "merge"):
                    uncontrolled_bounds = np.append(uncontrolled_bounds, np.array([[-500, 500], [-500, 500], [-velocity_bound, velocity_bound], [-velocity_bound, velocity_bound]]), axis=0)
                elif self.environment == "intersection":
                    uncontrolled_bounds = np.append(uncontrolled_bounds, np.array([[-100, 100], [-100, 100], [-velocity_bound, velocity_bound], [-velocity_bound, velocity_bound]]), axis=0)
                else: assert(False)

                if tmp_vehicle_num in group:
                    controlled_vars = controlled_vars + u_tmp
                    controlled_bounds = np.append(controlled_bounds, np.array([[-acceleration_bound, acceleration_bound], [-acceleration_bound, acceleration_bound]]), axis=0)
                else:
                    uncontrolled_vars = uncontrolled_vars + u_tmp
                    uncontrolled_bounds = np.append(uncontrolled_bounds, np.array([[-acceleration_bound, acceleration_bound], [-acceleration_bound, acceleration_bound]]), axis=0)

            uncontrolled_vars = tmp_contract.set_deter_uncontrolled_vars(uncontrolled_vars, bounds = uncontrolled_bounds)
            controlled_vars = tmp_contract.set_controlled_vars(controlled_vars, bounds = controlled_bounds)
            #  print(uncontrolled_vars)
            #  print(uncontrolled_bounds)
            #  print(controlled_vars)
            #  print(controlled_bounds)

            # Find the assumptions and guarantees formula of the contract
            # Initialize assumptions and guarantees
            
            # Find assumptions
            noncooperating_vehicle_num = len(self.vehicles) - len(group)
            
            if noncooperating_vehicle_num == 0:
                assumptions_formula = 'True'
            else:
                assumptions_condition = []
                for tmp_vehicle_num in self.vehicles:
                    if tmp_vehicle_num not in group:
                        assumptions_condition.append("({}_{} == 0)".format(self.control_names[0], tmp_vehicle_num))
                        assumptions_condition.append("({}_{} == 0)".format(self.control_names[1], tmp_vehicle_num))
                assumptions_formula = "(G[0,{}] ({}))".format(self.horizon, " & ".join(assumptions_condition))

            # Find guarantees
            # Find no collision guarantees
            vehicle_width = data["vehicle"]["width"]
            vehicle_length = data["vehicle"]["length"]
            if len(self.vehicles) <= 1:
                guarantees_formula = 'True'
            else:
                guarantees_condition = []
                # No collision between cooperating vehicles
                for (vehicle_num_1, vehicle_num_2) in combinations(self.vehicles, 2):
                    if (vehicle_num_1 in group) or (vehicle_num_2 in group):
                        if self.environment == "intersection":
                            condition = []
                            condition.append("({}_{} - {}_{} >= {})".format(self.state_names[0], vehicle_num_1, self.state_names[0], vehicle_num_2, 1.5*vehicle_width))
                            condition.append("({}_{} - {}_{} >= {})".format(self.state_names[0], vehicle_num_2, self.state_names[0], vehicle_num_1, 1.5*vehicle_width))
                            condition.append("({}_{} - {}_{} >= {})".format(self.state_names[1], vehicle_num_1, self.state_names[1], vehicle_num_2, 1.5*vehicle_width))
                            condition.append("({}_{} - {}_{} >= {})".format(self.state_names[1], vehicle_num_2, self.state_names[1], vehicle_num_1, 1.5*vehicle_width))
                            guarantees_condition.append("({})".format(" | ".join(condition)))
                        else:
                            condition = []
                            condition.append("({}_{} - {}_{} >= {})".format(self.state_names[0], vehicle_num_1, self.state_names[0], vehicle_num_2, 2*vehicle_length))
                            condition.append("({}_{} - {}_{} >= {})".format(self.state_names[0], vehicle_num_2, self.state_names[0], vehicle_num_1, 2*vehicle_length))
                            condition.append("({}_{} - {}_{} >= {})".format(self.state_names[1], vehicle_num_1, self.state_names[1], vehicle_num_2, 2*vehicle_width))
                            condition.append("({}_{} - {}_{} >= {})".format(self.state_names[1], vehicle_num_2, self.state_names[1], vehicle_num_1, 2*vehicle_width))
                            guarantees_condition.append("({})".format(" | ".join(condition)))
                guarantees_formula = "(G[0,{}] ({}))".format(self.horizon, " & ".join(guarantees_condition))


            # Set the contracts
            #  print(assumptions_formula)
            #  print(guarantees_formula)
            #  input()
            tmp_contract.set_assume(assumptions_formula)
            tmp_contract.set_guaran(guarantees_formula)

            # Saturate and add the contract
            tmp_contract.checkSat()
            tmp_model.add_contract(tmp_contract)

            # Add the contract specifications
            tmp_model.add_hard_constraint(tmp_contract.assumption)
            tmp_model.add_hard_constraint(tmp_contract.guarantee)

            # Region constraints
            # 3-D constraint: [vehicle_num, time, region_num]
            region_constraint = [[[] for t in range(self.horizon + 1)] for v in range(len(group))]
            for vehicle_id, vehicle_num in enumerate(group):
                region_params = np.array(data["vehicle"]["region"]["equation"][vehicle_num])
                region_params[np.abs(region_params) < EPS] = 0

                ego_x_var_name = "{}_{}".format(self.state_names[0], vehicle_num)
                ego_y_var_name = "{}_{}".format(self.state_names[1], vehicle_num)

                for t in range(self.horizon + 1):
                    for region_param in region_params:
                        region_formula = "(G[{},{}] (({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0) & ({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0) & ({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0) & ({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0)))".format(t, t, 
                                    region_param[0][0], ego_x_var_name, region_param[0][1], ego_x_var_name, region_param[0][2], ego_y_var_name, region_param[0][3], ego_y_var_name, region_param[0][4],
                                    region_param[1][0], ego_x_var_name, region_param[1][1], ego_x_var_name, region_param[1][2], ego_y_var_name, region_param[1][3], ego_y_var_name, region_param[1][4],
                                    region_param[2][0], ego_x_var_name, region_param[2][1], ego_x_var_name, region_param[2][2], ego_y_var_name, region_param[2][3], ego_y_var_name, region_param[2][4],
                                    region_param[3][0], ego_x_var_name, region_param[3][1], ego_x_var_name, region_param[3][2], ego_y_var_name, region_param[3][3], ego_y_var_name, region_param[3][4])
                        index = tmp_model.add_soft_constraint(region_formula)
                        region_constraint[vehicle_id][t].append(index)

            #  print(region_constraint)
            #  print(tmp_model.soft_constraints)

            # Set Dynamics
            for tmp_vehicle_num in self.vehicles: 
                # Find the vector of states and controls
                tmp_x = [tmp_contract.var("{}_{}".format(state_name, tmp_vehicle_num)) for state_name in self.state_names]
                tmp_x = Vector(tmp_x)

                tmp_u = [tmp_contract.var("{}_{}".format(control_name, tmp_vehicle_num)) for control_name in self.control_names]
                tmp_u = Vector(tmp_u)
                # Build a linear system dynamics

                # Add dynamics
                tmp_model.add_dynamic(Next(tmp_x) == A * tmp_x + B * tmp_u)
                #  print(Next(tmp_x) == A * tmp_x + B * tmp_u)
                #  input()

            # Add variables and constraints to MILP solver
            tmp_model.preprocess()
            
            # print(tmp_contract.deter_var_name2id)
            # print(tmp_solver.hard_constraints)
            # print(tmp_solver.soft_constraints)
            # print(tmp_solver.model.getVars())
            # print(tmp_model.soft_constraints_var)

            # Add region constraints
            for vehicle_id, vehicle_num in enumerate(group):
                for t in range(self.horizon + 1):
                    # Add a binary variable for regions at time t
                    tmp_model.model_add_binary_variable_by_name("regions_{}_{}".format(vehicle_num, t))
                    tmp_model.model.update()

                    var = [tmp_model.soft_constraints_var[i] for i in region_constraint[vehicle_id][t]]
                    tmp_model.model.addConstr(tmp_model.model.getVarByName("regions_{}_{}".format(vehicle_num, t)) == gp.or_(var))
                    tmp_model.model.addConstr(tmp_model.model.getVarByName("regions_{}_{}".format(vehicle_num, t)) == 1)
                    tmp_model.model.update()

            # Add objective to MILP sovler
            objective_func = gp.LinExpr()

            # Goal objectives
            for vehicle_num in group:
                for t in range(self.horizon + 1):
                    # Find the gurobi variable
                    position_var = [tmp_model.variable(tmp_contract.var("{}_{}".format(name, vehicle_num)).idx, t) for name in self.state_names[:2]]
                    #  print(position_var)
                    #  input()

                    # Add goal objectives in x and y axis
                    goal_x = tmp_model.model_add_continuous_variable_by_name("goal_x_{}_{}".format(vehicle_num, t), lb = -M, ub = M)
                    goal_y = tmp_model.model_add_continuous_variable_by_name("goal_y_{}_{}".format(vehicle_num, t), lb = -M, ub = M)
                    tmp_model.model.addConstr(position_var[0] - data["vehicle"]["target"][vehicle_num][0] <= goal_x)
                    tmp_model.model.addConstr(data["vehicle"]["target"][vehicle_num][0] - position_var[0] <= goal_x)
                    tmp_model.model.addConstr(position_var[1] - data["vehicle"]["target"][vehicle_num][1] <= goal_y)
                    tmp_model.model.addConstr(data["vehicle"]["target"][vehicle_num][1] - position_var[1] <= goal_y)
                    tmp_model.model.update()

                    # Add goal objective in x and y axis
                    objective_func += (len(self.vehicles)-vehicle_num) * goal_x + (len(self.vehicles)-vehicle_num) * goal_y

            # Fuel objectives (Sum of absolute values of u)
            for vehicle_num in group:
                for t in range(self.horizon + 1):
                    # Find the gurobi variable
                    control_var = [tmp_model.variable(tmp_contract.var("{}_{}".format(name, vehicle_num)).idx, t) for name in self.control_names]

                    # Add fuel objectives in x and y axis
                    fuel_x = tmp_model.model_add_continuous_variable_by_name("fuel_x_{}_{}".format(vehicle_num, t), lb = 0, ub = acceleration_bound)
                    fuel_y = tmp_model.model_add_continuous_variable_by_name("fuel_y_{}_{}".format(vehicle_num, t), lb = 0, ub = acceleration_bound)
                    tmp_model.model.addConstr(control_var[0] <= fuel_x)
                    tmp_model.model.addConstr(-control_var[0] <= fuel_x)
                    tmp_model.model.addConstr(control_var[1] <= fuel_y)
                    tmp_model.model.addConstr(-control_var[1] <= fuel_y)
                    tmp_model.model.update()

                    # Add fuel objective in x and y axis
                    objective_func += fuel_x + fuel_y

            #  # Add region objectives
            #  for t in range(self.horizon + 1):
            #     # Add a binary variable for regions at time t
            #     tmp_model.model_add_binary_variable_by_name("regions_{}".format(t))
            #
            #     # Add a constraint
            #     region_exprs = []
            #     for i in range(len(region_params)):
            #         for soft_const in tmp_model.soft_constraints_var:
            #             if i == soft_const[1] and t == soft_const[2]:
            #                 region_exprs.append(soft_const[0])
            #     tmp_model.model.addConstr(tmp_model.model.getVarByName("regions_{}".format(t)) == gp.or_(region_exprs))
            #     tmp_model.model.update()
            #
            #     # Add region objective
            #     objective_func += 10*M*tmp_model.model.getVarByName("regions_{}".format(t))

            # print(objective_func)
            # input()
            
            # Set objectives
            tmp_model.set_objective(objective_func)

            # Add the model to the model dictionary
            self.model_dict[tuple(group)] = tmp_model
            #  print(self.model_dict)
            #  input()

    def find_control(self, current_states):
        """ Find control for all the groups of cooperating vehicles (or a vehicle if only one in the group).

        :param init: [description]
        :type init: [type]
        """

        control = [[] for v in self.vehicles]
        status = True
        # Initialize output
        for group_idx, group_model in self.model_dict.items():

            group_contract = group_model.contract
            
            # Add the constraints for the current states
            for vehicle_num in self.vehicles:
                for state_var_idx in range(len(self.state_names)):
                    var = group_model.variable(group_contract.var("{}_{}".format(self.state_names[state_var_idx], vehicle_num)).idx, 0)
                    group_model.model.addConstr(var == current_states[vehicle_num][state_var_idx], 'state_{}_{}'.format(vehicle_num, state_var_idx))
            group_model.model.update()

            # # Setup for finding the adversarial controls
            # group_model.set_objective(group_model.model.getObjective(), sense='maximize')
            # for vehicle_num in vehicles:
            #     for t in range(self.horizon):
            #         for i in range(2):
            #             tmp_var_num = group_model.contract.deter_var_name2id["{}_{}".format(self.control_names[i], vehicle_num)]
            #             tmp_var_name = "contract_{}_{}".format(tmp_var_num, t)
            #             group_model.model.addConstr(group_model.model.getVarByName(tmp_var_name) == 0, 'adversarial_{}_{}'.format(tmp_var_num, t))
            # group_model.model.update()

            # # Solve and print the solution for the adversarial control                
            # solved = group_model.solve()
            # if solved:
            #     group_model.print_solution()

            # # Fetch the adversarial controls
            # for vehicle_num in self.vehicles:
            #     if str(vehicle_num) not in vehicles:
            #         tmp_output = group_model.fetch_control([var + "_" + str(vehicle_num) for var in self.control_names], length=self.horizon)
            #         adversarial_control[vehicle_num] = tmp_output

            # # Delete the constraints for the adversarial controls
            # for vehicle_num in vehicles:
            #     for t in range(self.horizon):
            #         for i in range(2):
            #             tmp_var_num = group_model.contract.deter_var_name2id["{}_{}".format(self.control_names[i], vehicle_num)]
            #             tmp_var_name = "contract_{}_{}".format(tmp_var_num, t)
            #             group_model.model.remove(group_model.model.getConstrByName('adversarial_{}_{}'.format(tmp_var_num, t)))
            # group_model.model.update()

            # print(adversarial_control)
            # input()

            # # Setup for finding the control
            # group_model.set_objective(group_model.model.getObjective())     
            # for adversarial_vehicle_num, adversarial_vehicle_control in adversarial_control.items():
            #     for t in range(self.horizon):
            #         for i in range(2):
            #             tmp_var_num = group_model.contract.deter_var_name2id["{}_{}".format(self.control_names[i], adversarial_vehicle_num)]
            #             tmp_var_name = "contract_{}_{}".format(tmp_var_num, t)
            #             group_model.model.addConstr(group_model.model.getVarByName(tmp_var_name) == adversarial_vehicle_control[i][t], 'adversarial_{}_{}'.format(tmp_var_num, t))
            # group_model.model.update()

            # Solve and fetch the solution for the control   
            solved = group_model.solve()
            if solved:
                group_model.print_solution()
                #  position_var = [group_model.variable(group_contract.var("{}_{}".format(name, 0)).idx, 30) for name in self.state_names[:2]]
                #  print(position_var[0].x)
                #  print(position_var[1].x)
                #  print(group_model.model.getVarByName("goal_y_1_30").x)

            # Fetch the control
            if solved:
                for vehicle_num in group_idx:
                    variables = [group_contract.var("{}_{}".format(var, vehicle_num)) for var in self.control_names]
                    tmp_output = group_model.fetch_solution(variables)
                    control[vehicle_num] = tmp_output
            else:
                for vehicle_num in group_idx:
                    control[vehicle_num] = [[0.0], [0.0]]
            status &= solved

            # # Delete the constraints for the control
            # for adversarial_vehicle_num in adversarial_control.keys():
            #     for t in range(self.horizon):
            #         for i in range(2):
            #             tmp_var_num = group_model.contract.deter_var_name2id["{}_{}".format(self.control_names[i], adversarial_vehicle_num)]
            #             tmp_var_name = "contract_{}_{}".format(tmp_var_num, t)
            #             group_model.model.remove(group_model.model.getConstrByName('adversarial_{}_{}'.format(tmp_var_num, t)))
            # group_model.model.update()

            # print(control)
            # input()

            # Delete the constraints for the current states
            for vehicle_num in self.vehicles:
                for state_var_idx in range(len(self.state_names)):
                    group_model.model.remove(group_model.model.getConstrByName('state_{}_{}'.format(vehicle_num, state_var_idx)))
            group_model.model.update()

        #  print(control)
        #  print(status)
        #  input()
        #  input()

        return control, not status
    
    def plot_path(self, param):
        """ Plots the pre-determined paths of the vehicles.
        """
        # Find region parameters
        param = np.array(param)

        # Initialize plots and settings
        if len(param) == 1:
            ax = [plt.gca()]
        elif self.environment in ("highway", "merge"):
            _, ax = plt.subplots(len(param), 1)
        elif self.environment == "intersection":
            _, ax = plt.subplots(1, len(param))
        else: assert(False)

        cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

        # For each region of a vehicle, find vertices and plot the region
        for vehicle_num in self.vehicles:
            
            im = None
            for region_param in param[vehicle_num]:
                f1 = lambda x,y : region_param[0,0]*x**2 + region_param[0,1]*x + region_param[0,2]*y**2 + region_param[0,3]*y + region_param[0,4]
                f2 = lambda x,y : region_param[1,0]*x**2 + region_param[1,1]*x + region_param[1,2]*y**2 + region_param[1,3]*y + region_param[1,4]
                f3 = lambda x,y : region_param[2,0]*x**2 + region_param[2,1]*x + region_param[2,2]*y**2 + region_param[2,3]*y + region_param[2,4]
                f4 = lambda x,y : region_param[3,0]*x**2 + region_param[3,1]*x + region_param[3,2]*y**2 + region_param[3,3]*y + region_param[3,4]

                if self.environment == "highway":
                    x = np.linspace(-10,300,1000)
                    y = np.linspace(-5,20,1000)
                elif self.environment == "merge":
                    x = np.linspace(0,550,1000)
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
            ax[vehicle_num].imshow(im, extent=(x.min(),x.max(),y.min(),y.max()), origin="lower", cmap=cmaps[vehicle_num])
                
        plt.savefig("test_{}.png".format(self.environment))
