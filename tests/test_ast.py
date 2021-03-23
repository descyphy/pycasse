import sys
sys.path.append("/home/kevin/Github/pystl")
from pystl.parser import Parser

p = Parser()
f = p("(& (> a 0) (<= b 0) (> c 0))")
print(repr(f))
print(f)

