import sys
sys.path.append("/home/kevin/Github/pystl")
from pystl import *
from pystl import Parser
import numpy as np

c1 = contract('c1')                   # Create a contract c1
[x] = c1.set_deter_uncontrolled_vars(['x'], dtypes = ['BINARY']) # Set a deterministic uncontrolled variable
[y] = c1.set_controlled_vars(['y'], bounds = np.array([[0,5]]))         # Set a controlled variable
[z] = c1.set_nondeter_uncontrolled_vars(['z'], np.array([0]), np.array([[1]]))         # Set a controlled variable

#  print(And(x**2>0, y> 0, z>0))
#  print((x>0) & (y> 0) & (z>0))
#  print((x>0).Until([0,2], (y> 0)))
#  input()
#  print((x>0) | true)
#  print((x>0) | false)
#  print((x>0) & true)
#  print((x>0) & false)
#  input()

p = Parser(c1)
#  print(p("3", "int"))
#  print(p("y", "variable"))
#  print(p("y** 3", "variable_power"))
#  print(p("3x", "const_variable"))
#  print(p("y**2", "const_variable"))
#  print(p("3x + 2y + 2z", "expression_outer"))
#  print(p("(2y + 4 - 2 > 3)", "AP"))
#  print(p("(P[0.1] (3x - 2y + 2z < 3))", "stAP"))
#  print(p("(G[1,3] (P[0.85] (2y <= 8)))"))
#  print(p("(G[0,3] (P[0.85] (2y + 4 - 2 > 3)))"))
#  print(p("[3,2]", "interval"))
#  print(p("(G[2,3] (x + 2 < 2))", "g"))
#  print(p("((y<2) R[2,3] (x + 2 < 2))", "r"))
#  print(p("(y<2) | (x + 2 < 2)", "or_inner"))
#  print(p("((y<2) | (x + 2 < 2))", "or_outer"))
#  print(p("(False & (x <=2))", "and_outer"))
#  print(p("((y<=0) -> (x <= 0) -> (z <= 0))", "implies_outer"))
#  print(p("True", "true"))
#  print(p("!(x>0)"))
#  print(p("((x == 10000.0) & (y == 8.0))", "and_outer"))
#  print(p("((y<2) & (x + 2 < 2))", "and_outer"))
#  print(p("(F[0,10] ((x == 10000.0) & (y == 8.0)))", "f"))
print(p("(F[0,10] ((x == 10000.0) & (y**2 + y**1 == 8.0)))"))
input()
