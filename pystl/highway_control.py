import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt

from pystl.variable import M, EPS
from pystl.vector import Vector, Next
from pystl.core import MILPSolver
from pystl.contracts import contract

def highway_openloop(data, H, init=None, savepath=True):
    """ Create a set of contracts for the vehicles on the highway environment.

    :param data: [description]
    :type data: [type]
    """
    # Fetch the environment name
    env = data["scenario"]

    # Find dynamics
    dt = float(data["dynamics"]["dt"])
    A = np.array(data["dynamics"]["A"])
    B = np.array(data["dynamics"]["B"])
    A = np.where(A=="dt", dt, A)
    B = np.where(B=="dt", dt, B)
    A = A.astype(float)
    B = B.astype(float)
    
    # Plot the path
    if savepath:
        # Find region parameters
        param = np.array(data["vehicle"]["region"]["equation"])

        # Initialize plots and settings
        if env in ("highway", "merge"):
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

                if env == "highway":
                    x = np.linspace(-10,300,1000)
                    y = np.linspace(-5,20,1000)
                elif env == "merge":
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
                
        plt.savefig("test_{}.png".format(env))

    # Initialize output
    output = []

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

        # Find the assumptions and guarantees formula of the contract
        # Initialize assumptions and guarantees
        assumptions_formula = "(G[0,{}] (".format(H)
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

        # # Find goal guarantees
        # x_goal = data["vehicle"]["target"][vehicle_num][0]
        # if x_goal >= M:
        #     x_goal = 300
        # y_goal = data["vehicle"]["target"][vehicle_num][1]
        # guarantees_formula += "(F[0,{}] (({} == {}) & ({} == {})))".format(H, "{}_{}".format(data["dynamics"]["x"][0], vehicle_num), x_goal, "{}_{}".format(data["dynamics"]["x"][1], vehicle_num), y_goal)
        
        # Find no collision guarantees
        vehicle_width = data["vehicle"]["width"]
        vehicle_length = data["vehicle"]["length"]
        if len(data["vehicle"]["id"]) >= 2:
            for i in data["vehicle"]["id"]:
                if i != vehicle_num:
                    if env == "intersection":
                        guarantees_formula += "(G[0,{}] (({} - {} >= {}) | ({} - {} >= {}) | ({} - {} >= {}) | ({} - {} >= {}))) & ".format(H, 
                                                "{}_{}".format(data["dynamics"]["x"][0], vehicle_num), "{}_{}".format(data["dynamics"]["x"][0], i), 4, 
                                                "{}_{}".format(data["dynamics"]["x"][0], i), "{}_{}".format(data["dynamics"]["x"][0], vehicle_num), 4, 
                                                "{}_{}".format(data["dynamics"]["x"][1], vehicle_num), "{}_{}".format(data["dynamics"]["x"][1], i), 4, 
                                                "{}_{}".format(data["dynamics"]["x"][1], i), "{}_{}".format(data["dynamics"]["x"][1], vehicle_num), 4)
            
                    else:
                        guarantees_formula += "(G[0,{}] (({} - {} >= {}) | ({} - {} >= {}) | ({} - {} >= {}) | ({} - {} >= {}))) & ".format(H, 
                                                "{}_{}".format(data["dynamics"]["x"][0], vehicle_num), "{}_{}".format(data["dynamics"]["x"][0], i), vehicle_length, 
                                                "{}_{}".format(data["dynamics"]["x"][0], i), "{}_{}".format(data["dynamics"]["x"][0], vehicle_num), vehicle_length, 
                                                "{}_{}".format(data["dynamics"]["x"][1], vehicle_num), "{}_{}".format(data["dynamics"]["x"][1], i), vehicle_width, 
                                                "{}_{}".format(data["dynamics"]["x"][1], i), "{}_{}".format(data["dynamics"]["x"][1], vehicle_num), vehicle_width)
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
        tmp_solver.add_contract(tmp_contract)

        # Add the contract specifications
        tmp_solver.add_hard_constraint(tmp_contract.assumption)
        tmp_solver.add_hard_constraint(tmp_contract.guarantee)

        # Region constraints
        region_params = np.array(data["vehicle"]["region"]["equation"][vehicle_num])
        region_params[np.abs(region_params) < EPS] = 0
        ego_vars = uncontrolled_vars[6*vehicle_num:6*vehicle_num+4]
        ego_x_var_name = ego_vars[0].name
        ego_y_var_name = ego_vars[1].name

        if len(region_params) == 1:
            region_param = region_params[0]
            for t in range(H):
                region_formula ="(G[{},{}] (({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0) & ({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0) & ({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0) & ({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0)))".format(t, t,
                               region_param[0][0], ego_x_var_name, region_param[0][1], ego_x_var_name, region_param[0][2], ego_y_var_name, region_param[0][3], ego_y_var_name, region_param[0][4],
                               region_param[1][0], ego_x_var_name, region_param[1][1], ego_x_var_name, region_param[1][2], ego_y_var_name, region_param[1][3], ego_y_var_name, region_param[1][4],
                               region_param[2][0], ego_x_var_name, region_param[2][1], ego_x_var_name, region_param[2][2], ego_y_var_name, region_param[2][3], ego_y_var_name, region_param[2][4],
                               region_param[3][0], ego_x_var_name, region_param[3][1], ego_x_var_name, region_param[3][2], ego_y_var_name, region_param[3][3], ego_y_var_name, region_param[3][4])
                tmp_solver.add_soft_constraint(region_formula, region_num=0, time=t)
        else:
            count = 0
            for region_param in region_params:
                for t in range(H):
                    region_formula = "(G[{},{}] (({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0) & ({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0) & ({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0) & ({} {}**2 + {} {} + {} {}**2 + {} {} + {} <= 0)))".format(t, t, 
                                region_param[0][0], ego_x_var_name, region_param[0][1], ego_x_var_name, region_param[0][2], ego_y_var_name, region_param[0][3], ego_y_var_name, region_param[0][4],
                                region_param[1][0], ego_x_var_name, region_param[1][1], ego_x_var_name, region_param[1][2], ego_y_var_name, region_param[1][3], ego_y_var_name, region_param[1][4],
                                region_param[2][0], ego_x_var_name, region_param[2][1], ego_x_var_name, region_param[2][2], ego_y_var_name, region_param[2][3], ego_y_var_name, region_param[2][4],
                                region_param[3][0], ego_x_var_name, region_param[3][1], ego_x_var_name, region_param[3][2], ego_y_var_name, region_param[3][3], ego_y_var_name, region_param[3][4])
                    tmp_solver.add_soft_constraint(region_formula, region_num=count, time=t)
                count += 1

        # Set initial states
        checked_ego = False
        for tmp_vehicle_num in data["vehicle"]["id"]:
            for j in range(4):
                if j == 0:
                    tmp_solver.add_hard_constraint(uncontrolled_vars[tmp_contract.deter_var_name2id["d_x_{}".format(tmp_vehicle_num)]-1] == init[tmp_vehicle_num][0])
                elif j == 1:
                    tmp_solver.add_hard_constraint(uncontrolled_vars[tmp_contract.deter_var_name2id["d_y_{}".format(tmp_vehicle_num)]-1] == init[tmp_vehicle_num][1])
                elif j == 2:
                    tmp_solver.add_hard_constraint(uncontrolled_vars[tmp_contract.deter_var_name2id["v_x_{}".format(tmp_vehicle_num)]-1] == init[tmp_vehicle_num][2])
                elif j == 3:
                    tmp_solver.add_hard_constraint(uncontrolled_vars[tmp_contract.deter_var_name2id["v_y_{}".format(tmp_vehicle_num)]-1] == init[tmp_vehicle_num][3])
        
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
            tmp_solver.add_dynamic(Next(tmp_x) == A * tmp_x + B * tmp_u)

        ## Set objectives
        # Initialize objective
        # objective_func = 0


        # Add variables and constraints to MILP solver
        tmp_solver.preprocess()
        
        # print(tmp_contract.deter_var_name2id)
        # print(tmp_solver.hard_constraints)
        # print(tmp_solver.soft_constraints)
        # print(tmp_solver.model.getVars())
        # input()

        # Add region constraint
        for t in range(H):
            # Add a binary variable for regions at time t
            tmp_solver.model_add_binary_variable_by_name("regions_{}".format(t))

            # Add a constraint
            region_exprs = []
            for i in range(len(region_params)):
                for soft_const in tmp_solver.soft_constraints_var:
                    if i == soft_const[1] and t == soft_const[2]:
                        region_exprs.append(soft_const[0])
            tmp_solver.model.addConstr(tmp_solver.model.getVarByName("regions_{}".format(t)) == 1)
            tmp_solver.model.addConstr(tmp_solver.model.getVarByName("regions_{}".format(t)) == gp.or_(region_exprs))
            tmp_solver.model.update()

        # Add objective to MILP sovler
        objective_func = 0

        # Goal objectives
        uncontrolled_vars[6*tmp_vehicle_num:6*tmp_vehicle_num+4]
        for i in range(H):
            tmp_solver.model_add_continuous_variable_by_name("goal_x_{}".format(i), lb = -M, ub = M)
            tmp_solver.model_add_continuous_variable_by_name("goal_y_{}".format(i), lb = -M, ub = M)
            tmp_solver.model.addConstr(tmp_solver.model.getVarByName("contract_{}_{}".format(uncontrolled_vars[6*vehicle_num].idx, i)) - data["vehicle"]["target"][vehicle_num][0] <= tmp_solver.model.getVarByName("goal_x_{}".format(i)))
            tmp_solver.model.addConstr(data["vehicle"]["target"][vehicle_num][0] - tmp_solver.model.getVarByName("contract_{}_{}".format(uncontrolled_vars[6*vehicle_num].idx, i)) <= tmp_solver.model.getVarByName("goal_x_{}".format(i)))
            tmp_solver.model.addConstr(tmp_solver.model.getVarByName("contract_{}_{}".format(uncontrolled_vars[6*vehicle_num+1].idx, i)) - data["vehicle"]["target"][vehicle_num][1] <= tmp_solver.model.getVarByName("goal_y_{}".format(i)))
            tmp_solver.model.addConstr(data["vehicle"]["target"][vehicle_num][1] - tmp_solver.model.getVarByName("contract_{}_{}".format(uncontrolled_vars[6*vehicle_num+1].idx, i)) <= tmp_solver.model.getVarByName("goal_y_{}".format(i)))
            tmp_solver.model.update()

            # Add goal objective in x and y axis
            objective_func += M*tmp_solver.model.getVarByName("goal_x_{}".format(i)) + M*tmp_solver.model.getVarByName("goal_y_{}".format(i))

        # Fuel objectives (Sum of absolute values of u)
        tmp_solver.model.write("MILP.lp")
        for i in range(H):
            # Find fule objectives in x and y axis
            tmp_solver.model_add_continuous_variable_by_name("fuel_x_{}".format(i), lb = 0, ub = acceleration_bound)
            tmp_solver.model_add_continuous_variable_by_name("fuel_y_{}".format(i), lb = 0, ub = acceleration_bound)
            tmp_solver.model.addConstr(tmp_solver.model.getVarByName("contract_{}_{}".format(controlled_vars[0].idx, i)) <= tmp_solver.model.getVarByName("fuel_x_{}".format(i)))
            tmp_solver.model.addConstr(-tmp_solver.model.getVarByName("contract_{}_{}".format(controlled_vars[0].idx, i)) <= tmp_solver.model.getVarByName("fuel_x_{}".format(i)))
            tmp_solver.model.addConstr(tmp_solver.model.getVarByName("contract_{}_{}".format(controlled_vars[1].idx, i)) <= tmp_solver.model.getVarByName("fuel_y_{}".format(i)))
            tmp_solver.model.addConstr(-tmp_solver.model.getVarByName("contract_{}_{}".format(controlled_vars[1].idx, i)) <= tmp_solver.model.getVarByName("fuel_y_{}".format(i)))
            tmp_solver.model.update()

            # Add fuel objective in x and y axis
            objective_func += tmp_solver.model.getVarByName("fuel_x_{}".format(i)) + tmp_solver.model.getVarByName("fuel_y_{}".format(i))
            
        # # Add region objectives
        # for t in range(H):
        #     # Add a binary variable for regions at time t
        #     tmp_solver.model_add_binary_variable_by_name("regions_{}".format(t))

        #     # Add a constraint
        #     region_exprs = []
        #     for i in range(len(region_params)):
        #         for soft_const in tmp_solver.soft_constraints_var:
        #             if i == soft_const[1] and t == soft_const[2]:
        #                 region_exprs.append(soft_const[0])
        #     tmp_solver.model.addConstr(tmp_solver.model.getVarByName("regions_{}".format(t)) == gp.or_(region_exprs))
        #     tmp_solver.model.update()
            
        #     # Add region objective
        #     objective_func += M*tmp_solver.model.getVarByName("regions_{}".format(t))

        # print(objective_func)
        # input()
        
        # Set objectives
        tmp_solver.set_objective(objective_func)

        # Solve the problem using MILP solver
        solved = tmp_solver.solve()
        if solved:
            tmp_solver.print_solution()

        # Fetch control output
        tmp_output = tmp_solver.fetch_control(controlled_vars)
        output.append(tmp_output)

    return output
