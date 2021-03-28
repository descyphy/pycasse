import sys
sys.path.append("/home/kevin/Github/pystl")
from pystl.parser import Parser, Expression

#  p = Parser()
#  f = p("(& (> a 0) (<= b 3) (> c 0))")
e1 = Expression("x1 + x2 + 3 x3+ 5 a")
e2 = Expression("245 - 4a")
print(e2)
print(e1 - e2)
#  e.add(None, 3)
#  print(e)
#  print(e * -3)

