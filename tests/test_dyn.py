import sys
from pystl import *
from pystl.parser import *
import numpy as np
import time

# Build a contract
c = contract('c')                                                  # Create a contract c
[x1, x2] = c.set_deter_uncontrolled_vars(['x1', 'x2'], \
        bounds = np.array([[-100, 2000], [-5, 10]]))               # Set a deterministic uncontrolled variable
[u1] = c.set_controlled_vars(['u1'], bounds = np.array([[-1, 1]])) # Set a controlled variable
c.set_assume('True')                                               # Set/define the assumptions
c.set_guaran('(F[0,30] (x1 >= 245))')                              # Set/define the guarantees
#  c.set_guaran('(F[0,100] (x[0] => 945))') # Set/define the guarantees
#  c.set_guaran('(F[0,200] (x[0] => 1945))') # Set/define the guarantees
#  c.set_guaran('(G[0,10] (F[0,30] (G[0,10] (x[0] => 245))))') # Set/define the guarantees
#  c.set_guaran('(G[0,10] ((F[0,5] (x[0] => 3)) & (F[0,5] (x[0] <= 0))))') # Set/define the guarantees
c.checkSat()  # Saturate c
c.printInfo()                                       # Print c2

# Build a linear system dynamics
x = Vector([x1, x2])
u = Vector([u1])
A = np.array([[1, 1], [0, 1]])
B = np.array([[0], [1]])

solver = MILPSolver()
solver.add_contract(c)
#  solver.add_dynamic(x * A + u * B == (Next(x) - x)/0.5)
solver.add_dynamic(Next(x) == A * x + B * u)
solver.add_constraint(x1 == 0)
solver.add_constraint(x2 == 0)
solver.add_constraint(c.guarantee)

# Solve the problem using MILP solver
start = time.time()
solved = solver.solve()
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
if solved:
    solver.print_solution()

# Build a SMC Solver for SSF
#  SSFsolver = SMCSolver()
#  SSFsolver.add_contract(c)
#  SSFsolver.add_dynamics(sys_dyn)
#
#  # Solve the problem using SMC solver
#  start = time.time()
#  solved = SSFsolver.solve("SSF")
#  end = time.time()
#  print("Time elaspsed for SMC (SSF): {} [seconds].\n".format(end - start))
# if solved:
# 	for v in SSFsolver.main_convex_solver.getVars():
# 		if 'x' in v.varName or 'u' in v.varName:
# 			print('%s %g' % (v.varName, v.x))

# Build a SMC Solver for IIS
#  IISsolver = SMCSolver()
#  IISsolver.add_contract(c)
#  IISsolver.add_dynamics(sys_dyn)
#
#  # Solve the problem using SMC solver
#  start = time.time()
#  solved = IISsolver.solve("IIS")
#  end = time.time()
#  print("Time elaspsed for SMC (IIS): {} [seconds].\n".format(end - start))
# if solved:
# 	for v in IISsolver.main_convex_solver.getVars():
# 		if 'x' in v.varName or 'u' in v.varName:
# 			print('%s %g' % (v.varName, v.x))
