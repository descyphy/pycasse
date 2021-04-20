import sys, os
from pystl import *
import numpy as np

c1 = contract('c1')                         # Create a contract c1
[x] = c1.set_deter_uncontrolled_vars(['x']) # Set a deterministic uncontrolled variable
[y] = c1.set_controlled_vars(['y'])         # Set a controlled variable

c1.set_assume('(G[0,3] (x => 5))')
c1.set_guaran('(G[0,4] (y => 2))')
c1.saturate()                               # Saturate c1
c1.printInfo()

c1.checkCompat(print_sol=True)              # Check compatibility of c1
c1.checkConsis(print_sol=True)              # Check consistency of c1
c1.checkFeas(print_sol=True)                # Check feasiblity of c1

# c2 = contract('c2')                         # Create a contract c2
# [x] = c2.set_deter_uncontrolled_vars(['x']) # Set a deterministic uncontrolled variable
# [y] = c2.set_controlled_vars(['y'])         # Set a controlled variable

# c2.set_assume('(F[1,2] (x => 4))')
# c2.set_guaran('(G[0,4] (y => 3))')
# c2.saturate()                               # Saturate c2
# #  c2.printInfo()

# c2.checkCompat()                            # Check compatibility of c2
# c2.checkConsis()                            # Check consistency of c2
# c2.checkFeas()                              # Check feasiblity of c2

# c3 = contract('c3')                         # Create a contract c3
# [x] = c3.set_deter_uncontrolled_vars(['x']) # Set a deterministic uncontrolled variable
# [y] = c3.set_controlled_vars(['y'])         # Set a controlled variable

# c3.set_assume('(F[1,2] (x => 4))')
# c3.set_guaran('(G[0,4] (y => 1))')
# c3.saturate()                               # Saturate c3
# #  c3.printInfo()

# c3.checkCompat()                            # Check compatibility of c3
# c3.checkConsis()                            # Check consistency of c3
# c3.checkFeas()                              # Check feasiblity of c3

# c2.checkRefine(c1, print_sol=True)          # Check whether c2 refines c1
# c3.checkRefine(c1, print_sol=True)          # Check whether c3 refines c1
