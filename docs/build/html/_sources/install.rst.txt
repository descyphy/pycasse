Installing PyCASSE
==================

Pre-requisites
--------------
* Install dependencies::

   $ pip3 install -r requirements.txt


Optimization Solver
-------------------
Gurobi
^^^^^^
PyCASSE uses Gurobi [Gurobi]_ for solving mixed integer programs (MIP).

* Install `Gurobi <https://www.gurobi.com/>`_
* Setup `Gurobi Python Interface for Python Users <https://www.gurobi.com/documentation/9.0/quickstart_mac/the_grb_python_interface_f.html>`_
* Get a Gurobi license
   * Faculty, students, or staff of an academic institution might be eligible for `a free academic license <https://www.gurobi.com/downloads/end-user-license-agreement-academic/>`_

.. CPLEX
.. ^^^^^
.. We plan to give CPLEX [Cplex]_ as an option for solving mixed integer linear programs (MILP) in near future.

PyCASSE
-------
* Download or clone the PyCASSE-git repository::

   $ git clone https://github.com/descyphy/pycasse

* Navigate to PyCASSE-git folder where ``setup.py`` is located

* Install PyCASSE::

   $ pip3 install .

You are now ready to use PyCASSE!
