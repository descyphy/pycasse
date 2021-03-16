from pystl import *
import numpy as np
import time

# Build a contract
c = contract('c') # Create a contract c
c.set_assume('True') # Set/define the assumptions
c.set_guaran('(F[0,30] (=> x[0] 245))') # Set/define the guarantees
# c.set_guaran('(F[0,100] (=> x[0] 945))') # Set/define the guarantees
# c.set_guaran('(F[0,200] (=> x[0] 1945))') # Set/define the guarantees
# c.set_guaran('(G[0,10] (F[0,30] (G[0,10] (=> x[0] 245))))') # Set/define the guarantees
# c.set_guaran('(G[0,10] (& (F[0,5] (=> x[0] 3)) (F[0,5] (<= x[0] 0))))') # Set/define the guarantees
c.saturate()  # Saturate c
c.printInfo() # Print c

# Build a linear system dynamics
x_len = 2
u_len = 1
x_bounds = np.array([[-100, 2000], [-5, 10]])
u_bounds = np.array([[-1, 1]])
x0 = np.array([[0], [0]])
A = np.array([[1, 1], [0, 1]])
B = np.array([[0], [1]])
sys_dyn = lin_dyn(x_len=x_len, u_len=u_len, x_bounds=x_bounds, u_bounds=u_bounds, x0=x0, A=A, B=B)

# Build a MILP Solver
MILPsolver = MILPSolver()
MILPsolver.add_contract(c)
MILPsolver.add_dynamics(sys_dyn)

# Solve the problem using MILP solver
start = time.time()
solved = MILPsolver.solve()
end = time.time()
print("Time elaspsed for MILP: {} [seconds].\n".format(end - start))
# if solved:
# 	for v in MILPsolver.MILP_convex_solver.getVars():
# 		if 'x' in v.varName or 'u' in v.varName:
# 			print('%s %g' % (v.varName, v.x))

# Build a SMC Solver for SSF
SSFsolver = SMCSolver()
SSFsolver.add_contract(c)
SSFsolver.add_dynamics(sys_dyn)

# Solve the problem using SMC solver
start = time.time()
solved = SSFsolver.solve("SSF")
end = time.time()
print("Time elaspsed for SMC (SSF): {} [seconds].\n".format(end - start))
# if solved:
# 	for v in SSFsolver.main_convex_solver.getVars():
# 		if 'x' in v.varName or 'u' in v.varName:
# 			print('%s %g' % (v.varName, v.x))

# Build a SMC Solver for IIS
IISsolver = SMCSolver()
IISsolver.add_contract(c)
IISsolver.add_dynamics(sys_dyn)

# Solve the problem using SMC solver
start = time.time()
solved = IISsolver.solve("IIS")
end = time.time()
print("Time elaspsed for SMC (IIS): {} [seconds].\n".format(end - start))
# if solved:
# 	for v in IISsolver.main_convex_solver.getVars():
# 		if 'x' in v.varName or 'u' in v.varName:
# 			print('%s %g' % (v.varName, v.x))