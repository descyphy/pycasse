Control Synthesis with A/G Contract
===================================

PyCASSE enables reasoning about the behavior of a component :math:`M`, with given dynamics, using A/G contracts.
In this chapter, we sythesize the control trajectory of a component with linear dynamics such that it implements a contract :math:`C`, i.e., :math:`M \models C`.

Control Synthesis with STL A/G Contract in PyCASSE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: figs/dyn_stl_setup.png
   :width: 200
   :align: center

Assume a point-mass robot on a line where its location, velocity, and acceleration are denoted as :math:`s`, :math:`v`, and :math:`a`.

.. figure:: figs/dyn_stl.png
   :width: 300
   :align: center

The dynamics of the point-mass robot :math:`M` on a line is given as :math:`x_{k+1} = A x_k + B u_k` where :math:`x_k = [s_k, v_k]^T` and :math:`u_k = [a_k]`.
The goal is to synthesize the control input :math:`\mathbf{u}^{15} = u_0 u_1 \ldots u_{14}` where :math:`\forall k, u_k \in [-1, 1]` such that 
the robot visits the regions :math:`\{ r | r \geq 3\}` and :math:`\{ r | r \leq 0\}` every :math:`5~s`, i.e., :math:`\phi_G := \mathbf{G}_{[0,10]} ((\mathbf{F}_{[0,5]} (s \geq 3)) \land (\mathbf{F}_{[0,5]} (s \leq 0)))`, is satisfied.

Given the initial state :math:`x_0 = [0, 0]^T`, PyCASSE can synthesize the control input by running the following Python script:

.. code-block:: python

   from pycasse import *
   import matplotlib.pyplot as plt
   import time

   # Build a contract
   c = contract('c')                                               # Create a contract c
   c.add_deter_vars(['s', 'v', 'a'], 
      bounds = [[-100, 1000], [-5, 10], [-1, 1]])                  # Set deterministic variables
   c.set_assume('True')                                            # Set/define the assumptions
   c.set_guaran('G[0,10] ((F[0,5] (s => 3)) & (F[0,5] (s <= 0)))') # Set/define the guarantees
   c.checkSat()                                                    # Saturate c
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
      plt.savefig('test_dyn_stl_result_fig.pdf')

.. figure:: figs/dyn_stl_result.png
   :width: 750
   :align: center

The figure on the left shows the trajectory of the robot's location; 
the figure on the center shows the trajectory of the robot's velocity; 
and the figure on the right shows the trajectory of the robot's acceleration (control input).
As shown in the above figure, the control input synthesized by PyCASSE guarantees the satisfaction of the STL specification :math:`\phi_G`.

For details, refer to :download:`test_dyn_stl.py <../../tests/stl_tests/test_dyn_stl.py>`.

Control Synthesis with StSTL A/G Contract in PyCASSE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's assume the same point-mass robot on a line from the previous example, 
but this time, its plant is subject to Gaussian uncertainty :math:`w \sim Gaussin([0, 0]^T, [[0, 0], [0, 0.5^2]])`.

.. figure:: figs/dyn_ststl.png
   :width: 300
   :align: center

The dynamics of the point-mass robot :math:`M'` on a line is given as :math:`x_{k+1} = A x_k + B u_k + w_k` where :math:`x_k = [s_k, v_k]^T` and :math:`u_k = [a_k]`.
The goal is to synthesize the control input :math:`\mathbf{u}^{10} = u_0 u_1 \ldots u_9` where :math:`\forall k, u_k \in [-1, 1]` such that 
the robot eventually in :math:`10~s` reaches the region :math:`\{ r | r \geq 34\}` with probability larger than or equal to :math:`0.9`, i.e., :math:`\phi'_G := \mathbf{F}_{[0,10]} (\mathbb{P} \{ s \geq 34 \} \geq 0.9)`, is satisfied.

Given the initial state :math:`x_0 = [0, 0]^T`, PyCASSE can synthesize the control input by running the following Python script:

.. code-block:: python

   from pycasse import *
   import matplotlib.pyplot as plt
   import time

   # Build a contract
   c_prime = contract('c_prime')                    # Create a contract c_prime
   c_prime.add_deter_vars(['s', 'v', 'a'], 
      bounds = [[-100, 2000], [-5, 10], [-1, 1]])   # Set deterministic variables
   c_prime.set_assume('True')                       # Set/define the assumptions
   c_prime.set_guaran('F[0,10] (P[0.9] (s => 34))') # Set/define the guarantees
   c_prime.checkSat()                               # Saturate c_prime
   c_prime.printInfo()                              # Print c_prime

   # Initialize a milp solver and add a contract
   solver = MILPSolver()
   solver.add_contract(c_prime)

   # Build a linear system dynamics
   solver.add_dynamics(x = ['s', 'v'], u = ['a'], A = [[1, 1], [0, 1]], B = [[0], [1]], Q = [[0, 0], [0, 0.5**2]])

   # Add initial conditions
   solver.add_init_condition('s == 0')
   solver.add_init_condition('v == 0')

   # Add guarantee constraints
   solver.add_constraint(c_prime.guarantee, name='b_g')

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

.. figure:: figs/dyn_ststl_result.png
   :width: 300
   :align: center

The above figure shows the trajectory of the robot's acceleration (control input) synthesized by PyCASSE.
To validate whether the the synthesized control input satisfies the StSTL specification :math:`\phi'_G`, :math:`10^5` MATLAB simulations were ran by using the synthesized control input. 

.. figure:: figs/dyn_ststl_simu.png
   :width: 350
   :align: center

   :math:`10^5` simulation in MATLAB.

The above figure shows that the control input synthesized by PyCASSE satisfies the StSTL specification :math:`\phi'_G`.
For details, refer to :download:`test_dyn_ststl.py <../../tests/ststl_tests/test_dyn_ststl.py>` and :download:`test_dyn_ststl_simu.m <../../tests/ststl_tests/test_dyn_ststl_simu.m>`.