import sys
sys.path.append("/home/kevin/Github/pystl")
from pystl import *
# from pystl import Parser
from pystl.parser import ststl_grammar, Parser
import numpy as np

# c1 = contract('c1')                   # Create a contract c1
# [x] = c1.set_deter_uncontrolled_vars(['x'], dtypes = ['BINARY']) # Set a deterministic uncontrolled variable
# [y] = c1.set_controlled_vars(['y'], bounds = np.array([[0,5]]))         # Set a controlled variable
# [z] = c1.set_nondeter_uncontrolled_vars(['z'], np.array([0]), np.array([[1]]))         # Set a controlled variable

# print(And(x**2>0, y> 0, z>0))
#  print((x>0) & (y> 0) & (z>0))
#  print((x>0).Until([0,2], (y> 0)))
#  input()
#  print((x>0) | true)
#  print((x>0) | false)
#  print((x>0) & true)
#  print((x>0) & false)
#  input()
# tree = ststl_grammar.parse("P[x] (-3*x*y^2*z^3 + x >= x2*y^5-2)")
# print(tree)
# print(ststl_grammar.parse("(F[1,3] (x >= 0)) & (P[x] (3*x*y^2*z + x >= x2*y^5-2)) & (P[x] (3*x*y^2*z^3 + x >= x2*y^5-2))"))

p = Parser()
# [ast, formula] = p("!(True)")
# [ast, formula] = p("!(3*x*y^2*z^3 + x => x2*y^5 - 2)")
[ast, formula] = p("X(F[0,25] (3*x*y^2*z^3 + x => x2*y^5 - 2))")
# [ast, formula] = p("!((F[0,25](3*x*y^2*z^3 + x < x2*y^5 + 3)) U[0,12] (P[x] (3*x*y^2*z^3 + x => x2*y^5 - 2)))")
# [ast, formula] = p("(F[0,10] (3*x*y^2*z^3 + x => x2*y^5 - 2)) -> (G[1,12] (P[x] (3*x*y^2*z^3 + x => x2*y^5 - 2)))")
# [ast, formula] = p("!((F[0,10] (P[x] (3*x*y^2*z^3 + x => x2*y^5 - 2))) -> (G[1,12] (P[x] (3*x*y^2*z^3 + x => x2*y^5 - 2))))")
# [ast, formula] = p("!((F[0,10] (P[x] (3*x*y^2*z^3 + x => x2*y^5 - 2))) & (-3.3*x*y33^2*z^3 - x^6*y + 1 > z) & (G[0,10] (3*x*y^2*z^3 + x => x2*y^5 - 2)))")
print(ast[0].find_horizon(0,0))
ast[0].printInfo()
negated = ast[0].push_negation()
negated.printInfo()
print(negated)

# output = p("3.3", "multiterm")
# output = p("-8 - 3.3*x*y33^2*z^3 - x^6 + 9", "expression")
# output = p("-3.3*x*y33^2*z^3 - x^6*y + 1 > z", "AP")
# output = p("P[x + y] (3*x*y^2*z^3 + x => x2*y^5 + 5)", "stAP")
# output = p("(-3.3*x*y33^2*z^3 - x^6*y + 1 > z) -> (G[0,10] (P[x] (3*x*y^2*z^3 + x => x2*y^5 - 2)))", "nontemporal_binary")
# [ast, formula] = p("(!(True)) & (-3.3*x*y33^2*z^3 - x^6*y + 1 > z) & (G[0,10] (P[x] (3*x*y^2*z^3 + x => x2*y^5 - 2)))")
# output = p("(F[0,10] (P[x] (3*x*y^2*z^3 + x => x2*y^5 - 2))) & (-3.3*x*y33^2*z^3 - x^6*y + 1 > z) & (G[0,10] (P[x] (3*x*y^2*z^3 + x => x2*y^5 - 2)))")

#  #  print(p("3", "int"))
#  print(p("y", "variable"))
#  print(p("y** 3", "variable_power"))
#  print(p("3x", "const_variable"))
#  print(p("y**2", "const_variable"))
#  print(p("3 y**2", "const_variable"))
# print(p("3*x**2 - 2*y**3 + 2*z", "expression"))
#  print(p("(2*y*y2 + 4 - 2 > 3)", "AP"))
#  print(p("(P[0.1] (3x - 2y + 2z < 3))", "stAP"))
# print(p("G[1,3] (P[0.85] (2*y <= 8))"))
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
# print(p("(F[0,10] ((x == 10000.0) & (y**2 + y**1 == 8.0)))"))
