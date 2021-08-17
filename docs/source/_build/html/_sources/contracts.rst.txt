Assume-Guarantee (A/G) Contracts
================================

What is an A/G Contract?
------------------------
An assume-guarantee (A/G) contract :math:`C` is a triple :math:`(V,A,G)` where :math:`V` is the set of variables, :math:`A` is the set of behaviors which a component (or system) expects from its environment, and :math:`G` is the set of behaviors that the component promises given that the environment provides behaviors within the assumption set. We say :math:`M` is an `implementation` of :math:`C` by writing :math:`M \models C`.

A contract is in `saturated` form if it satisfies :math:`\overline{A} \subseteq G` where :math:`\overline{A}` is the complement of :math:`A`. An unsaturated contract :math:`C'=(V,A,G)` can always be `saturated` to the equivalent contract :math:`C=(V,A,G')` where :math:`G'=G \cup \overline{A}`.

An A/G contract :math:`C = (V,A,G)` is `compatible` if there exists an environment behavior over the set of variables :math:`V`, i.e., :math:`A \neq \emptyset`, `consistent` if there exists a behavior satisfying the (saturated) guarantees, i.e., :math:`G \neq \emptyset`, and `feasible` if there exists a behavior that satisfies both the assumptions and the guarantees, i.e., :math:`A \cap G \neq \emptyset`.

Via `refinement`, we can reason about different abstraction layers in system design. A contract :math:`C_2 = (V,A_2,G_2)` refines :math:`C_1 = (V,A_1,G_1)` if and only if (1) :math:`A_1 \subseteq A_2` and (2) :math:`G_2 \subseteq G_1`. We denote this relationship as :math:`C_2 \preceq C_1`. Intuitively, if a contract :math:`C_2` refines another contract :math:`C_1`, then :math:`C_2` can replace :math:`C_1`. For further details we refer to the monograph [ContractMono]_.

A/G Contracts in PySTL
----------------------

Creating an A/G Contract
^^^^^^^^^^^^^^^^^^^^^^^^
.. image:: system1.png
   :width: 300
   :align: center

Consider the component above. An STL contract :math:`C_1 = (V_1,A_1,G_1)` where :math:`V_1 := \{ x, y \}`, :math:`A_1 := \mathbf{G}_{[0,3]}(x \geq 5)`, and :math:`G_1 := \mathbf{G}_{[1,4]}(y \geq 2)` can be created as follows:

.. code-block:: python

   from pystl import *
   import numpy as np

   c1 = contract('c1')                   # Create a contract c1
   c1.set_deter_uncontrolled_vars(['x']) # Set a deterministic uncontrolled variable
   c1.set_controlled_vars(['y'])         # Set a controlled variable

   c1.set_assume('(G[0,3] (x => 5))')    # Set assumptions of c1
   c1.set_guaran('(G[1,3] (y => 2))')    # Set guarantees of c1
   c1.checkSat()                         # Saturate c1
   c1.printInfo()                        # Print information of c1

.. image:: system2.png
   :width: 300
   :align: center

Using StSTL specifications, a contract :math:`C_1' = (V_1', A_1', G_1')` where :math:`V_1' = U' \cup X_1'`, :math:`U_1' := \{ w_1, w_2 \}`, :math:`X_1' := \{ x, y \}`, :math:`A_1' := \mathbf{G}_{[0,3]}(5 \leq x)`, :math:`G_1' := \mathbf{G}_{[1,3]}(P\{ y-2w_1+3w_2 \leq 8 \} \geq 0.95)`, and :math:`\mathbf{w} = [w_1, w_2]^T \sim N([0,2]^T, [[1,0],[0,1]])` can be created as follows:

.. code-block:: python

   c1_prime = contract('c1')                                      # Create a contract c1_prime
   c1_prime.set_deter_uncontrolled_vars(['x'])                    # Set a deterministic uncontrolled variable
   c1_prime.set_nondeter_uncontrolled_vars(['w1', 'w2'], \
         mean = np.array([0, 2]), cov = np.array([[1**2, 0], [0, 1**2]]))
                                                                  # Set nondeterministic uncontrolled variables
   c1_prime.set_controlled_vars(['y'])                            # Set a controlled variable
   c1_prime.set_assume('(G[0,3] (5 <= x))')                       # Set assumptions of c1_prime
   c1_prime.set_guaran('(G[1,3] (P[0.85] (y - 2w1 + 3w2 <= 8)))') # Set guarantees of c1_prime
   c1_prime.printInfo()                                           # Print c1_prime

Any contract in PySTL can be saturated and its information can be printed. For example, :math:`C_1` and :math:`C_1'` can be saturated and their information can be printed as follows:

.. code-block:: python

   c1.saturate()                         # Saturate c1
   c1.printInfo()                        # Print c1

   c1_prime.saturate()                   # Saturate c1_prime
   c1_prime.printInfo()                  # Print c1_prime

Please note that in PySTL, contracts are automatically saturated whenever necessary. 

Checking Compatibility, Consistency, and Feasibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We can check `compatibility`, `consistency`, and `feasiblity` of the contract :math:`C_1`. This includes when PySTL checks for `compatibility`, `consistency`, and `feasiblity`.

.. code-block:: python

   c1.checkCompat(print_sol=True)          # Check compatibility of c1
   c1.checkConsis(print_sol=True)          # Check consistency of c1
   c1.checkFeas(print_sol=True)            # Check feasiblity of c1

Refinement
^^^^^^^^^^
Let's create two more contracts :math:`C_2` and :math:`C_3` and check their `compatibility`, `consistency`, and `feasiblity`:

.. code-block:: python

   c2 = contract('c2')                   # Create a contract c2
   c2.set_assume('F[1,2](x>=4)')         # Set an assumption
   c2.set_guaran('G[0,4](y>=3)')         # Set a guarantee such that c2 refines c1
   c2.set_deter_uncontrolled_vars(['x']) # Set a deterministic noncontrolled variable
   c2.set_controlled_vars(['y'])         # Set a controlled variable
   c2.saturate()                         # Saturate c2
   c2.printInfo()                        # Print c2

.. code-block:: python

   c3 = contract('c3')                   # Create a contract c3
   c3.set_assume('F[1,2](x>=4)')         # Set an assumption
   c3.set_guaran('G[0,4](y>=0)')         # Set a guarantee such that c3 does not refines c1
   c3.set_deter_uncontrolled_vars(['x']) # Set a deterministic noncontrolled variable
   c3.set_controlled_vars(['y'])         # Set a controlled variable
   c3.saturate()                         # Saturate c3
   c3.printInfo()                        # Print c3

We can check whether :math:`C_2` and :math:`C_3` `refines` :math:`C_1` or not:

.. code-block:: python

   c1.checkRefine(c2)                   # Check whether c2 refines c1
   c1.checkRefine(c3)                   # Check whether c3 refines c1

Similarly, it is also possible to check refinement between probabilistic contracts. For an example with STL contracts refer to :download:`test_contracts_stl.py <../../tests/test_contracts_stl.py>`. For an example with StSTL contracts refer to :download:`test_contracts_ststl.py <../../tests/test_contracts_ststl.py>`.

PySTL Contract Class
^^^^^^^^^^^^^^^^^^^^
.. automodule:: pystl.contracts.contract
	:members:
	:exclude-members: reset_controlled_vars, reset_uncontrolled_vars, addVars2model, printSol

A/G Contracts Operations
------------------------
..
   A/G contracts :math:`C_1 = (V_1, A_1, G_1)` and :math:`C_2 = (V_2, A_2, G_2)` can be combined using contract operations: `conjunction` (:math:`\wedge`) (or `greatest lower bound` (:math:`\sqcap`)), `composition` (:math:`\otimes`), `merging` (:math:`\cdot`), and `least upper bound` (:math:`\sqcup`). A combined contract using these contract operations has the following properties:

A/G contracts :math:`C_1 = (V_1, A_1, G_1)` and :math:`C_2 = (V_2, A_2, G_2)` can be combined using contract operations: `conjunction` (:math:`\wedge`) and `composition` (:math:`\otimes`). A combined contract using these contract operations can be computed as follows:

.. math::

   C_1 \wedge C_2 = & \; (V_1 \cup V_2, A_1 \cup A_2, G_1 \cap G_2) \\
   C_1 \otimes C_2 = & \; (V_1 \cup V_2, (A_1 \cap A_2) \cup \overline{(G_1 \cap G_2)}, G_1 \cap G_2) 

An A/G contract :math:`C = (V, A, G)` can also be decomposed via the `quotient` operation. Given a contract :math:`C_2 = (V_2, A_2, G_2)` the quotient contract :math:`C/C_2`, has the following properties:

.. math::

   & C / C_2 \otimes C_2 \preceq C \\
   & \forall C_1, C_1 \otimes C_2 \preceq C \leftrightarrow C_1 \preceq C / C_2

The quotient can be computed as follows [Passerone19]_:

.. math::

   C / C_2 = (V \cup V_2, A \cap G_2, (G \cap A_2) \cup \overline{(A \cap G_2)})

..
   and that acquired from `separation` operation, denoted as :math:`C \div C_2`, has the properties:

   .. math::

      & \forall C_1, C \preceq C_1 \cdot C_2 \leftrightarrow C \div C_2 \preceq C_1 \\
      & C \div C_2 = (V \cup V_2, (A \cap G_2) \cup \overline{(G \cap A_2)}, G \cap A_2) \

A/G Contracts Operations in PySTL
---------------------------------

Combining Contracts
^^^^^^^^^^^^^^^^^^^
Assume that we have contracts :math:`C_1` and :math:`C_2`, we can combine them by using `conjunction` and `composition`.

.. code-block:: python

   c12_conj = conjunction(c1, c2)          # Conjunction of c1 and c2
   c12_conj.printInfo()                    # Print c12_conj

   c12_comp = composition(c1, c2)          # Composition of c1 and c2
   c12_comp.printInfo()                    # Print c12_comp

..
   c12_merge = merge(c1, c2)               # Merge of c1 and c2
   c12_merge.printInfo()                   # Print c12_merge

For an example where contracts are combined, refer to :download:`test_contracts_combine.py <../../tests/test_contracts_combine.py>`. 

Decomposing Contracts
^^^^^^^^^^^^^^^^^^^^^
Assume that we have contracts :math:`C` and :math:`C_2`. We can split :math:`C` into :math:`C_2` and :math:`C / C_2` by using `quotient` such that :math:`(C / C_2) \otimes C_2 \preceq C`.

.. code-block:: python

  c2_quo = quotient(c, c2)                # Quotient c/c2
  c2_quo.printInfo()                      # Print c2_quo

  c2_comp = composition(c2_quo, c2)       # Composition of c2_quo and c2
  c2_comp.printInfo()                     # Print c2_comp

  c.checkRefine(c2_comp)                  # Check whether c2_comp refines c

For an example where contracts are split using `quotient` operation, refer to :download:`test_contracts_quotient.py <../../tests/test_contracts_quotient.py>`. 


   We can also split :math:`C` into :math:`C_2` and :math:`C%C_2` by using `separation` such that :math:`C \preceq (C/C_2) \cdot C_2`.

.. .. code-block:: python

..    c2_sep = separation(c, c2)              # Separation c%c2
..    c2_sep.saturate()                       # Saturate c2_sep
..    c2_sep.printInfo()                      # Print c2_sep

..    c2_merge = merge(c2_sep, c2)            # Merge of c2_sep and c2
..    c2_merge.saturate()                     # Saturate c2_merge
..    c2_merge.printInfo()                    # Print c2_merge

..    c2_merge.checkRefine(c)                 # Check whether c refines c2_merge

.. For an example where contracts are split using `separation` operation, refer to :download:`test_contracts_separation.py <../../tests/test_contracts_separation.py>`.

PySTL Contract Operations
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pystl.contracts
	:members:
	:exclude-members: set_params, merge_contract_variables, quotient, separation
