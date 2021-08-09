import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
from numpy.lib.npyio import save
from six import iterkeys

from pystl.variable import M, EPS
from pystl.vector import Vector, Next
from pystl.core import MILPSolver
from pystl.contracts import contract

class highway_env_controller:
    __slots__ = ('environment', 'model_dict', 'vehicles', 'state_names', 'control_names', 'current_states', 'horizon')

    def __init__(self, highway_setup_data, horizon, debug = False):
        self.environment = ''
        self.model_dict = {}
        self.vehicles = []
        self.state_names = []
        self.control_names = []
        self.current_states = []
        self.horizon = 0

        self.load_setup(highway_setup_data, horizon)

    def load_setup(self, data, horizon, savepath=True):
        """ Loads the setup of the highway environment.

        :param data: [description]
        :type data: [type]
        """
        
        # Fetch information on the environment
        self.vehicles = data["vehicle"]["id"]
        self.environment = data["scenario"]
        self.horizon = horizon
        self.state_names = data["dynamics"]["x"]
        self.control_names = data["dynamics"]["u"]

        # Find dynamics
        dt = float(data["dynamics"]["dt"])
        A = np.array(data["dynamics"]["A"])
        B = np.array(data["dynamics"]["B"])
        A = np.where(A=="dt", dt, A).astype(float)
        B = np.where(B=="dt", dt, B).astype(float)

        # Plot and save the pre-determined paths of the vehicles in the environment
        if savepath:
            self.plot_path(data)

        # TODO: Modify to.. Add a model for each group of cooperating vehicles (or a vehicle)
        # Add a model for each vehicle
        for vehicle_num in data["vehicle"]["id"]:
            # Build a MILP solver
            tmp_model = MILPSolver()

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

            # Find the assumptions and guarantees formula of the contract
            # Initialize assumptions and guarantees
            assumptions_formula = "(G[0,{}] (".format(self.horizon)
            guarantees_formula = "("
            
            # Find assumptions
            total_vehicle_num = len(data["vehicle"]["id"])
            for i in data["vehicle"]["id"]:
                if i != vehicle_num:
                    assumptions_formula += "({}_{} == 0) & ({}_{} == 0)".format(data["dynamics"]["u"][0], i, data["dynamics"]["u"][1], i)
                    if i == total_vehicle_num-1 or (vehicle_num == total_vehicle_num-1 and i == vehicle_num-1):
                        assumptions_formula += "))"
                    else:
                        assumptions_formula += " & "

            # Find no collision guarantees
            vehicle_width = data["vehicle"]["width"]
            vehicle_length = data["vehicle"]["length"]
            if len(data["vehicle"]["id"]) >= 2:
                for i in data["vehicle"]["id"]:
                    if i != vehicle_num:
                        if self.environment == "intersection":
                            guarantees_formula += "(G[0,{}] (({} - {} >= {}) | ({} - {} >= {}) | ({} - {} >= {}) | ({} - {} >= {}))) & ".format(self.horizon, 
                                                    "{}_{}".format(data["dynamics"]["x"][0], vehicle_num), "{}_{}".format(data["dynamics"]["x"][0], i), 1.5*vehicle_width, 
                                                    "{}_{}".format(data["dynamics"]["x"][0], i), "{}_{}".format(data["dynamics"]["x"][0], vehicle_num), 1.5*vehicle_width, 
                                                    "{}_{}".format(data["dynamics"]["x"][1], vehicle_num), "{}_{}".format(data["dynamics"]["x"][1], i), 1.5*vehicle_width, 
                                                    "{}_{}".format(data["dynamics"]["x"][1], i), "{}_{}".format(data["dynamics"]["x"][1], vehicle_num), 1.5*vehicle_width)
                
                        else:
                            guarantees_formula += "(G[0,{}] (({} - {} >= {}) | ({} - {} >= {}) | ({} - {} >= {}) | ({} - {} >= {}))) & ".format(self.horizon, 
                                                    "{}_{}".format(data["dynamics"]["x"][0], vehicle_num), "{}_{}".format(data["dynamics"]["x"][0], i), 2*vehicle_length, 
                                                    "{}_{}".format(data["dynamics"]["x"][0], i), "{}_{}".format(data["dynamics"]["x"][0], vehicle_num), 2*vehicle_length, 
                                                    "{}_{}".format(data["dynamics"]["x"][1], vehicle_num), "{}_{}".format(data["dynamics"]["x"][1], i), 1.5*vehicle_width, 
                                                    "{}_{}".format(data["dynamics"]["x"][1], i), "{}_{}".format(data["dynamics"]["x"][1], vehicle_num), 1.5*vehicle_width)
                guarantees_formula = guarantees_formula[0:len(guarantees_formula)-3] + ')'
            else:
                guarantees_formula = guarantees_formula[1:-1]

            # Set the contracts
            tmp_contract.set_assume(assumptions_formula)
            tmp_contract.set_guaran(guarantees_formula)
            # print(assumptions_formula)
            # print(guarantees_formula)

            # Saturate contract
            tmp_contract.checkSat()
            # print(tmp_contract)
            
            # Add the contract
            tmp_model.add_contract(tmp_contract)

            # Add the contract specifications
            tmp_model.add_hard_constraint(tmp_contract.assumption)
            tmp_model.add_hard_constraint(tmp_contract.guarantee)

            # Region constraints
            region_params = np.array(data["vehicle"]["region"]["equation"][vehicle_num])
            region_params[np.abs(region_params) < EPS] = 0
            ego_vars = uncontrolled_vars[6*vehicle_num:6*vehicle_num+4]
            ego_x_var_name = ego_vars[0].name
            ego_y_var_name = ego_vars[1].name

            if len(region_params) == 1:
                region_param = region_params[0]
                for t in range(self.horizon):
                    region_formula ="(G[{},{}] (({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0) & ({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0) & ({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0) & ({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0)))".format(t, t,
                                region_param[0][0], ego_x_var_name, region_param[0][1], ego_x_var_name, region_param[0][2], ego_y_var_name, region_param[0][3], ego_y_var_name, region_param[0][4],
                                region_param[1][0], ego_x_var_name, region_param[1][1], ego_x_var_name, region_param[1][2], ego_y_var_name, region_param[1][3], ego_y_var_name, region_param[1][4],
                                region_param[2][0], ego_x_var_name, region_param[2][1], ego_x_var_name, region_param[2][2], ego_y_var_name, region_param[2][3], ego_y_var_name, region_param[2][4],
                                region_param[3][0], ego_x_var_name, region_param[3][1], ego_x_var_name, region_param[3][2], ego_y_var_name, region_param[3][3], ego_y_var_name, region_param[3][4])
                    tmp_model.add_soft_constraint(region_formula, region_num=0, time=t)
            else:
                count = 0
                for region_param in region_params:
                    for t in range(self.horizon):
                        region_formula = "(G[{},{}] (({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0) & ({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0) & ({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0) & ({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0)))".format(t, t, 
                                    region_param[0][0], ego_x_var_name, region_param[0][1], ego_x_var_name, region_param[0][2], ego_y_var_name, region_param[0][3], ego_y_var_name, region_param[0][4],
                                    region_param[1][0], ego_x_var_name, region_param[1][1], ego_x_var_name, region_param[1][2], ego_y_var_name, region_param[1][3], ego_y_var_name, region_param[1][4],
                                    region_param[2][0], ego_x_var_name, region_param[2][1], ego_x_var_name, region_param[2][2], ego_y_var_name, region_param[2][3], ego_y_var_name, region_param[2][4],
                                    region_param[3][0], ego_x_var_name, region_param[3][1], ego_x_var_name, region_param[3][2], ego_y_var_name, region_param[3][3], ego_y_var_name, region_param[3][4])
                        tmp_model.add_soft_constraint(region_formula, region_num=count, time=t)
                    count += 1

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

                # Add dynamics
                tmp_model.add_dynamic(Next(tmp_x) == A * tmp_x + B * tmp_u)

            # Add variables and constraints to MILP solver
            tmp_model.preprocess()
            
            # print(tmp_contract.deter_var_name2id)
            # print(tmp_solver.hard_constraints)
            # print(tmp_solver.soft_constraints)
            # print(tmp_solver.model.getVars())
            # input()

            # Add region constraint
            for t in range(self.horizon):
                # Add a binary variable for regions at time t
                tmp_model.model_add_binary_variable_by_name("regions_{}".format(t))

                # Add a constraint
                region_exprs = []
                for i in range(len(region_params)):
                    for soft_const in tmp_model.soft_constraints_var:
                        if i == soft_const[1] and t == soft_const[2]:
                            region_exprs.append(soft_const[0])
                tmp_model.model.addConstr(tmp_model.model.getVarByName("regions_{}".format(t)) == 1)
                tmp_model.model.addConstr(tmp_model.model.getVarByName("regions_{}".format(t)) == gp.or_(region_exprs))
                tmp_model.model.update()

            # Add objective to MILP sovler
            objective_func = M*tmp_model.model.getVarByName("node_0_0")

            # Goal objectives
            uncontrolled_vars[6*tmp_vehicle_num:6*tmp_vehicle_num+4]
            for i in range(self.horizon):
                tmp_model.model_add_continuous_variable_by_name("goal_x_{}".format(i), lb = -M, ub = M)
                tmp_model.model_add_continuous_variable_by_name("goal_y_{}".format(i), lb = -M, ub = M)
                tmp_model.model.addConstr(tmp_model.model.getVarByName("contract_{}_{}".format(uncontrolled_vars[6*vehicle_num].idx, i)) - data["vehicle"]["target"][vehicle_num][0] <= tmp_model.model.getVarByName("goal_x_{}".format(i)))
                tmp_model.model.addConstr(data["vehicle"]["target"][vehicle_num][0] - tmp_model.model.getVarByName("contract_{}_{}".format(uncontrolled_vars[6*vehicle_num].idx, i)) <= tmp_model.model.getVarByName("goal_x_{}".format(i)))
                tmp_model.model.addConstr(tmp_model.model.getVarByName("contract_{}_{}".format(uncontrolled_vars[6*vehicle_num+1].idx, i)) - data["vehicle"]["target"][vehicle_num][1] <= tmp_model.model.getVarByName("goal_y_{}".format(i)))
                tmp_model.model.addConstr(data["vehicle"]["target"][vehicle_num][1] - tmp_model.model.getVarByName("contract_{}_{}".format(uncontrolled_vars[6*vehicle_num+1].idx, i)) <= tmp_model.model.getVarByName("goal_y_{}".format(i)))
                tmp_model.model.update()

                # Add goal objective in x and y axis
                objective_func += tmp_model.model.getVarByName("goal_x_{}".format(i)) + tmp_model.model.getVarByName("goal_y_{}".format(i))

            # Fuel objectives (Sum of absolute values of u)
            for i in range(self.horizon):
                # Find fule objectives in x and y axis
                tmp_model.model_add_continuous_variable_by_name("fuel_x_{}".format(i), lb = 0, ub = acceleration_bound)
                tmp_model.model_add_continuous_variable_by_name("fuel_y_{}".format(i), lb = 0, ub = acceleration_bound)
                tmp_model.model.addConstr(tmp_model.model.getVarByName("contract_{}_{}".format(controlled_vars[0].idx, i)) <= tmp_model.model.getVarByName("fuel_x_{}".format(i)))
                tmp_model.model.addConstr(-tmp_model.model.getVarByName("contract_{}_{}".format(controlled_vars[0].idx, i)) <= tmp_model.model.getVarByName("fuel_x_{}".format(i)))
                tmp_model.model.addConstr(tmp_model.model.getVarByName("contract_{}_{}".format(controlled_vars[1].idx, i)) <= tmp_model.model.getVarByName("fuel_y_{}".format(i)))
                tmp_model.model.addConstr(-tmp_model.model.getVarByName("contract_{}_{}".format(controlled_vars[1].idx, i)) <= tmp_model.model.getVarByName("fuel_y_{}".format(i)))
                tmp_model.model.update()

                # Add fuel objective in x and y axis
                objective_func += tmp_model.model.getVarByName("fuel_x_{}".format(i)) + tmp_model.model.getVarByName("fuel_y_{}".format(i))
                
            # # Add region objectives
            # for t in range(self.horizon):
            #     # Add a binary variable for regions at time t
            #     tmp_model.model_add_binary_variable_by_name("regions_{}".format(t))

            #     # Add a constraint
            #     region_exprs = []
            #     for i in range(len(region_params)):
            #         for soft_const in tmp_model.soft_constraints_var:
            #             if i == soft_const[1] and t == soft_const[2]:
            #                 region_exprs.append(soft_const[0])
            #     tmp_model.model.addConstr(tmp_model.model.getVarByName("regions_{}".format(t)) == gp.or_(region_exprs))
            #     tmp_model.model.update()
                
            #     # Add region objective
            #     objective_func += 10*M*tmp_model.model.getVarByName("regions_{}".format(t))

            # print(objective_func)
            # input()
            
            # Set objectives
            tmp_model.set_objective(objective_func)

            # Add the model to the model dictionary
            # TODO: Modify when changed to a group of vehicles, e.g. "1_2_3" if vehicle 1, 2, 3 are cooperating
            self.model_dict["{}".format(vehicle_num)] = tmp_model

    def find_control(self, current_states):
        """ Find control for all the groups of cooperating vehicles (or a vehicle if only one in the group).

        :param init: [description]
        :type init: [type]
        """

        # Initialize output
        control = []

        for group_name, group_model in self.model_dict.items():
            # Initialize 
            # adversarial_control = {}
            vehicles = group_name.split('_')
            
            # Add the constraints for the current states
            for vehicle_num in self.vehicles:
                for state_var_idx in range(len(self.state_names)):
                    tmp_var_num = group_model.contract.deter_var_name2id["{}_{}".format(self.state_names[state_var_idx], vehicle_num)]
                    tmp_var_name = "contract_{}_0".format(tmp_var_num)
                    group_model.model.addConstr(group_model.model.getVarByName(tmp_var_name) == current_states[int(vehicle_num)][state_var_idx], 'state_{}_{}'.format(vehicle_num, state_var_idx))
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

            # Fetch the control
            for vehicle_num in vehicles:
                tmp_output, synthesis_fail = group_model.fetch_control([var + "_" + vehicle_num for var in self.control_names])
                control.append(tmp_output)

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

        return control, synthesis_fail
    
    def plot_path(self, data):
        """ Plots the pre-determined paths of the vehicles.
        """
        # Find region parameters
        param = np.array(data["vehicle"]["region"]["equation"])

        # Initialize plots and settings
        if self.environment in ("highway", "merge"):
            _, ax = plt.subplots(len(param), 1)
        else:
            _, ax = plt.subplots(1, len(param))
        cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

        # For each region of a vehicle, find vertices and plot the region
        for vehicle_num in data["vehicle"]["id"]:
            region_count = 0
            
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
                region_count += 1
            ax[vehicle_num].imshow(im, extent=(x.min(),x.max(),y.min(),y.max()), origin="lower", cmap=cmaps[vehicle_num])
                
        plt.savefig("test_{}.png".format(self.environment))