import sys
sys.path.append("/home/kevin/Github/pystl")
from pystl.variable import Var, DeterVar, NondeterVar
from pystl.parser import Expression

x = DeterVar('x', 1, 'controlled')
y = DeterVar('y', 2, 'controlled', bounds = np.array([0,5]))
z = NondeterVar('z', 0)
print(x)
print(y)
print(z)

print(3 > y)
print(-2 <= x + y + z + 1)
print(0 > x * 3 + y / 4 + z + 1)
print(x - y < 0)


#  print(u.__dict__)
