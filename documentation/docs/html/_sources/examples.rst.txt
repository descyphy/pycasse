Examples
========

.. Multi-Agent Control Synthesis using Reward Contracts
.. ----------------------------------------------------

.. Controller Synthesis in Highway Scenario
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Using StSTL specifications, a contract :math:`C_1' = (V_1', A_1', G_1')` where :math:`V_1' = U' \cup X_1'`, :math:`U_1' := \{ w_1, w_2 \}`, :math:`X_1' := \{ x, y \}`, :math:`A_1' := \mathbf{G}_{[0,3]}(5 \leq x)`, :math:`G_1' := \mathbf{G}_{[1,3]}(P\{ y-2w_1+3w_2 \leq 8 \} \geq 0.95)`, and :math:`\mathbf{w} = [w_1, w_2]^T \sim N([0,2]^T, [[1,0],[0,1]])` can be created as follows:

.. .. code-block:: python

..    c1_prime = contract('c1')                                      # Create a contract c1_prime
..    c1_prime.set_deter_uncontrolled_vars(['x'])                    # Set a deterministic uncontrolled variable
..    c1_prime.set_nondeter_uncontrolled_vars(['w1', 'w2'], \
..          mean = np.array([0, 2]), cov = np.array([[1**2, 0], [0, 1**2]]))
..                                                                   # Set nondeterministic uncontrolled variables
..    c1_prime.set_controlled_vars(['y'])                            # Set a controlled variable
..    c1_prime.set_assume('(G[0,3] (5 <= x))')                       # Set assumptions of c1_prime
..    c1_prime.set_guaran('(G[1,3] (P[0.85] (y - 2w1 + 3w2 <= 8)))') # Set guarantees of c1_prime
..    c1_prime.printInfo()                                           # Print c1_prime

.. Any contract in PyCASSE can be saturated and its information can be printed. For example, :math:`C_1` and :math:`C_1'` can be saturated and their information can be printed as follows:

.. Controller Synthesis in Highway Merging Scenario
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Using StSTL specifications, a contract :math:`C_1' = (V_1', A_1', G_1')` where :math:`V_1' = U' \cup X_1'`, :math:`U_1' := \{ w_1, w_2 \}`, :math:`X_1' := \{ x, y \}`, :math:`A_1' := \mathbf{G}_{[0,3]}(5 \leq x)`, :math:`G_1' := \mathbf{G}_{[1,3]}(P\{ y-2w_1+3w_2 \leq 8 \} \geq 0.95)`, and :math:`\mathbf{w} = [w_1, w_2]^T \sim N([0,2]^T, [[1,0],[0,1]])` can be created as follows:

.. .. code-block:: python

..    c1_prime = contract('c1')                                      # Create a contract c1_prime
..    c1_prime.set_deter_uncontrolled_vars(['x'])                    # Set a deterministic uncontrolled variable
..    c1_prime.set_nondeter_uncontrolled_vars(['w1', 'w2'], \
..          mean = np.array([0, 2]), cov = np.array([[1**2, 0], [0, 1**2]]))
..                                                                   # Set nondeterministic uncontrolled variables
..    c1_prime.set_controlled_vars(['y'])                            # Set a controlled variable
..    c1_prime.set_assume('(G[0,3] (5 <= x))')                       # Set assumptions of c1_prime
..    c1_prime.set_guaran('(G[1,3] (P[0.85] (y - 2w1 + 3w2 <= 8)))') # Set guarantees of c1_prime
..    c1_prime.printInfo()                                           # Print c1_prime

.. Any contract in PyCASSE can be saturated and its information can be printed. For example, :math:`C_1` and :math:`C_1'` can be saturated and their information can be printed as follows:


.. Controller Synthesis in Intersection Scenario
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Using StSTL specifications, a contract :math:`C_1' = (V_1', A_1', G_1')` where :math:`V_1' = U' \cup X_1'`, :math:`U_1' := \{ w_1, w_2 \}`, :math:`X_1' := \{ x, y \}`, :math:`A_1' := \mathbf{G}_{[0,3]}(5 \leq x)`, :math:`G_1' := \mathbf{G}_{[1,3]}(P\{ y-2w_1+3w_2 \leq 8 \} \geq 0.95)`, and :math:`\mathbf{w} = [w_1, w_2]^T \sim N([0,2]^T, [[1,0],[0,1]])` can be created as follows:

.. .. code-block:: python

..    c1_prime = contract('c1')                                      # Create a contract c1_prime
..    c1_prime.set_deter_uncontrolled_vars(['x'])                    # Set a deterministic uncontrolled variable
..    c1_prime.set_nondeter_uncontrolled_vars(['w1', 'w2'], \
..          mean = np.array([0, 2]), cov = np.array([[1**2, 0], [0, 1**2]]))
..                                                                   # Set nondeterministic uncontrolled variables
..    c1_prime.set_controlled_vars(['y'])                            # Set a controlled variable
..    c1_prime.set_assume('(G[0,3] (5 <= x))')                       # Set assumptions of c1_prime
..    c1_prime.set_guaran('(G[1,3] (P[0.85] (y - 2w1 + 3w2 <= 8)))') # Set guarantees of c1_prime
..    c1_prime.printInfo()                                           # Print c1_prime

.. Any contract in PyCASSE can be saturated and its information can be printed. For example, :math:`C_1` and :math:`C_1'` can be saturated and their information can be printed as follows:

Parameter Synthesis using StSTL Contracts (ICCAD 2022)
------------------------------------------------------

Finding Parameters for Multi-Sensor Perception system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The case study can be run::

   $ python tests/ICCAD2022/sensor_experiment.py

Finding Parameters for Cruise Control System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: figs/acc_fig1.png
   :width: 300
   :align: center

.. figure:: figs/acc_fig2.png
   :class: with-shadow
   :width: 300
   :align: center

The cruise control system in the figure above controls the ego vehicle to keep it as close as possible to a target distance :math:`d_{target} = d_{safe} + \tau v_e`, while adapting to the leading vehicle's behavior. :math:`d_{safe}` is the pre-determined safe distance and :math:`\tau` is the time gap. Several parts of such system are intrinsically of stochastic nature, e.g., the noise of the sensors detecting the distance and velocity, and the behavior of the leading vehicle. In this case study, we illustrate the parameter synthesis process on an system whose safety and comfort requirements are specified by two parametric stochastic contracts.

In this case study, we search for the sets of optimal parameter values :math:`\pi_{safe} = (c_s, p_s)` and :math:`\pi_{comf} = (c_c, p_c)` for two requirements expressed as the parametric stochastic contracts :math:`C_{safe}(\pi_{safe})` and :math:`C_{comf}(\pi_{comf})`. 

:math:`C_{safe}` requires that the probability of maintaining the distance :math:`d` larger than or equal to :math:`c_{s}` is greater than or equal to :math:`p_{s}` when the initial distance is greater than or equal to :math:`d_{target}` and the initial relative velocity between the ego and the leading vehicle is smaller than or equal to :math:`5~m/s`.

:math:`C_{safe} = (\phi_{A,safe}, \phi_{G,safe})` where :math:`\phi_{A,safe} := (d \geq d_{target}) \land (|v| \leq 5)` and :math:`\phi_{G,safe} := \mathbf{G}_{[0,20]} ( c_{s} - d )^{[p_{s}]}`

:math:`C_{comf}` requires that the acceleration of the ego vehicle be larger than or equal to :math:`c_{c}~m/s^2` with a probability larger than or equal to :math:`p_{c}`, to avoid abrupt decelerations under the same assumptions as :math:`C_{safe}`.

:math:`C_{comf} = (\phi_{A,comf}, \phi_{G,comf})` where :math:`\phi_{A,comf} := (d \geq d_{target}) \land (|v| \leq 5)` and :math:`\phi_{G,comf} := \mathbf{G}_{[0,20]} ( c_{c} - a_e ) ^{[p_{c}]}`

The case study can be run::

   $ python tests/ICCAD2022/ACC_experiment.py

As a result, feasible regions for finding the set of optimal parameter values :math:`\pi_{safe} = (c_s, p_s)`:

.. figure:: figs/acc_safety_result.png
   :width: 300
   :align: center

and the optimal paramter values is :math:`(c_{s}^*, p_{s}^*) = (5.625, 0.99375)`.

Similarly, feasible regions for finding the set of optimal parameter values :math:`\pi_{comf} = (c_c, p_c)` can be found:

.. figure:: figs/acc_comfort_result.png
   :class: with-shadow
   :width: 300
   :align: center

and the optimal paramter values is :math:`(c_{c}^*, p_{c}^*) = (-7.5, 0.99375)`.
