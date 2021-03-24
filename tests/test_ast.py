import sys
sys.path.append("/home/kevin/Github/pystl")
from pystl.parser import Parser, Expression

#  p = Parser()
#  f = p("(& (> a 0) (<= b 3) (> c 0))")
e = Expression("1 + a + 3+ 2b")
#  print(repr(e))
print(e)
e.add(None, 3)
print(e)
print(e * -3)

