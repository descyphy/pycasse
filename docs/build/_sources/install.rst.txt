Installing PySTL
=================

Pre-requisites
--------------
* Install dependencies::

   $ pip3 install -r requirements.txt


Optimization Solver
-------------------
Gurobi
^^^^^^
PySTL uses Gurobi [Gurobi]_ for solving mixed integer linear programs (MILP).

* Install `Gurobi <https://www.gurobi.com/>`_
* Setup `Gurobi Python Interface for Python Users <https://www.gurobi.com/documentation/9.0/quickstart_mac/the_grb_python_interface_f.html>`_
* Get a Gurobi license
   * Faculty, students, or staff of an academic institution might be eligible for `a free academic license <https://www.gurobi.com/downloads/end-user-license-agreement-academic/>`_

CPLEX
^^^^^
We plan to give CPLEX [Cplex]_ as an option for solving mixed integer linear programs (MILP) in near future.

PySTL
-----
* Download or clone the PySTL-git repository::

   $ git clone https://gitlab.com/descyphy/cyber-physical-system-group/deepcontracts/pystl

* Navigate to PySTL-git folder where ``setup.py`` is located

* Install PySTL::

   $ pip3 install .

You are now ready to use PySTL!
