import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ) )

from pystl import *
import numpy as np

c1 = contract('c1')                         # Create a contract c1
[x] = c1.set_deter_uncontrolled_vars(['x']) # Set a deterministic uncontrolled variable
[y] = c1.set_controlled_vars(['y'])         # Set a controlled variable

# c1.set_assume('(!(((5 <= x) & (x <= 9)) -> ((3 <= x) & (x <= 9))))')
c1.set_assume('((5 <= x) & (x <= 9))')
c1.set_guaran('(2 <= y)')
c1.checkSat()                               # Saturate c1
c1.printInfo()

#  c1.checkCompat(print_sol=True)              # Check compatibility of c1
#  c1.checkConsis(print_sol=True)              # Check consistency of c1
#  c1.checkFeas(print_sol=True)                # Check feasiblity of c1

c2 = contract('c2')                         # Create a contract c2
[x] = c2.set_deter_uncontrolled_vars(['x']) # Set a deterministic uncontrolled variable
[y] = c2.set_controlled_vars(['y'])         # Set a controlled variable

c2.set_assume('((3 <= x) & (x <= 9))')
c2.set_guaran('(3 <= y)')
c2.checkSat()                               # Saturate c2
c2.printInfo()

#  c2.checkCompat(print_sol=True)              # Check compatibility of c2
#  c2.checkConsis(print_sol=True)              # Check consistency of c2
#  c2.checkFeas(print_sol=True)                # Check feasiblity of c2

# c3 = contract('c3')                         # Create a contract c3
# [x] = c3.set_deter_uncontrolled_vars(['x']) # Set a deterministic uncontrolled variable
# [y] = c3.set_controlled_vars(['y'])         # Set a controlled variable

# c3.set_assume('(4 <= x)')
# c3.set_guaran('(1 <= y)')
# c3.checkSat()                               # Saturate c3
# c3.printInfo()

# c3.checkCompat(print_sol=True)              # Check compatibility of c3
# c3.checkConsis(print_sol=True)              # Check consistency of c3
# c3.checkFeas(print_sol=True)                # Check feasiblity of c3

c2.checkRefine(c1, print_sol=True)          # Check whether c2 refines c1
# c3.checkRefine(c1, print_sol=True)          # Check whether c3 refines c1
