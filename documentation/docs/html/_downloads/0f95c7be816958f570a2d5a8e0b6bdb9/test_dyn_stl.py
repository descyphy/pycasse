from pycasse import *
import matplotlib.pyplot as plt
import time

# Build a contract
c = contract('c')                                               # Create a contract c
c.add_deter_vars(['s', 'v', 'a'], 
    bounds = [[-100, 1000], [-5, 10], [-1, 1]])                 # Set deterministic variables
c.set_assume('True')                                            # Set/define the assumptions
c.set_guaran('G[0,10] ((F[0,5] (s => 3)) & (F[0,5] (s <= 0)))') # Set/define the guarantees
c.saturate()                                                    # Saturate c
c.printInfo()                                                   # Print c

# Initialize a milp solver and add a contract
solver = MILPSolver()
solver.add_contract(c)

# Build a linear system dynamics
solver.add_dynamics(x = ['s', 'v'], u = ['a'], A = [[1, 1], [0, 1]], B = [[0], [1]])

# Add initial conditions
solver.add_init_condition('s == 0')
solver.add_init_condition('v == 0')

# Add guarantee constraints
solver.add_constraint(c.guarantee, name='b_g')

# Solve the problem using MILP solver
start = time.time()
solved = solver.solve()
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))

# Print and plot the trajectory of the component/system
if solved:
    # Print the trajectory
    comp_traj = solver.print_solution()

    # Plot the trajectory
    fig, axs = plt.subplots(1,3)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    x = list(range(16))
    thres1 = [3]*16
    thres2 = [0]*16
    axs[0].plot(x, comp_traj['s'])
    axs[0].plot(x, thres1, 'r')
    axs[0].plot(x, thres2, 'r')
    axs[0].set_xlabel(r'$Time [s]$', fontsize='large')
    axs[0].set_ylabel(r'$s [m]$', fontsize='large')

    axs[1].plot(x, comp_traj['v'])
    axs[1].set_xlabel(r'$Time [s]$', fontsize='large')
    axs[1].set_ylabel(r'$v [m/s]$', fontsize='large')

    axs[2].plot(x, comp_traj['a'])
    axs[2].set_xlabel(r'$Time [s]$', fontsize='large')
    axs[2].set_ylabel(r'$a [m/s^2]$', fontsize='large')

    # Save the figure
    plt.savefig('test_dyn_stl_log_fig.pdf')