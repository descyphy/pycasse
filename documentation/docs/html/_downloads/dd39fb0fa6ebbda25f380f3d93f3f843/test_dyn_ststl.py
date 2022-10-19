from pycasse import *
import matplotlib.pyplot as plt
import time

# Build a contract
c = contract('c')                               # Create a contract c
c.add_deter_vars(['s', 'v', 'a'], 
    bounds = [[-100, 2000], [-5, 10], [-1, 1]]) # Set deterministic variables
c.set_assume('True')                            # Set/define the assumptions
c.set_guaran('F[0,10] (P[0.9] (s => 34))')      # Set/define the guarantees
c.checkSat()                                    # Saturate c
c.printInfo()                                   # Print c

# Initialize a milp solver and add a contract
solver = MILPSolver()
solver.add_contract(c)

# Build a linear system dynamics
solver.add_dynamics(x = ['s', 'v'], u = ['a'], A = [[1, 1], [0, 1]], B = [[0], [1]], Q = [[0, 0], [0, 0.5**2]])

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
    fig, ax = plt.subplots()
    x = list(range(11))

    ax.plot(x, comp_traj['a'])
    ax.set_xlabel(r'$Time [s]$', fontsize='large')
    ax.set_ylabel(r'$a [m/s^2]$', fontsize='large')

    # Save the figure
    plt.savefig('test_dyn_ststl_result_fig.pdf')
