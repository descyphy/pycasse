import sys
sys.path.append("/home/kevin/Github/pystl")
from pystl import *
from pystl import Parser
import numpy as np

c1 = contract('c1')                   # Create a contract c1
[x] = c1.set_deter_uncontrolled_vars(['x'], dtypes = ['BINARY']) # Set a deterministic uncontrolled variable
[y] = c1.set_controlled_vars(['y'], bounds = np.array([[0,5]]))         # Set a controlled variable
[z] = c1.set_nondeter_uncontrolled_vars(['z'], np.array([0]), np.array([[1]]))         # Set a controlled variable

#  c1.set_assume((x <= 3).U([0,3], x >= 5))    # Set/define the assumptions
#  c1.set_guaran(G([1,4], y >= 2))             # Set/define the guarantees
#  c1.set_assume("((x <= 3) U[0,3] (x >=5))")    # Set/define the assumptions
#  c1.set_guaran("(G[1,4] (y >= 2))")             # Set/define the guarantees

c1.saturate()                         # Saturate c1
print(c1)
input()

#  print((x>0) | true)
#  print((x>0) | false)
#  print((x>0) & true)
#  print((x>0) & false)
#  input()

p = Parser(c1)
#  print(p("x", "const_variable"))
#  print(p("y", "const_variable"))
#  print(p("3x + 2y + 2z", "expression"))
#  print(p("(2y / 4) - 2", "expression_outer"))
#  print(p("(2y + 4 - 2 > 3)", "AP"))
#  print(p("(P[3] (2y + 4 - 2 > 3))", "stAP"))
#  print(p("[3,2]", "interval"))
#  print(p("(G[2,3] (x + 2 < 2))", "g"))
#  print(p("((y<2) R[2,3] (x + 2 < 2))", "r"))
#  print(p("(y<2) | (x + 2 < 2)", "or_inner"))
#  print(p("((y<2) | (x + 2 < 2))", "or_outer"))
#  print(p("(False & (x <=2))", "and_outer"))
#  print(p("((y<=0) -> (x <= 0) -> (z <= 0))", "implies_outer"))
#  print(p("True", "true"))
#  print(p("!(x>0)"))
input()
