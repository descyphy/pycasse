from pystl import *
import numpy as np

c1 = contract('c1')                     # Create a contract c1
c1.set_deter_uncontrolled_vars(['x'])   # Set a deterministic uncontrolled variable
c1.set_controlled_vars(['y', 'z'])      # Set a controlled variable

c1.set_assume('((5 <= x) & (x <= 9))')  # Set assumptions of c1
c1.set_guaran('((2 <= y) & (z <= 2))')  # Set guarantees of c1
c1.checkSat()                           # Saturate c1
c1.printInfo()                          # Print information of c1

c1.checkCompat(print_sol=True)          # Check compatibility of c1
c1.checkConsis(print_sol=True)          # Check consistency of c1
c1.checkFeas(print_sol=True)            # Check feasiblity of c1
 
c2 = contract('c2')                     # Create a contract c2
c2.set_deter_uncontrolled_vars(['x'])   # Set a deterministic uncontrolled variable
c2.set_controlled_vars(['y'])           # Set a controlled variable

c2.set_assume('((6 <= x) & (x <= 9))')  # Set assumptions of c2
c2.set_guaran('(3 <= y)')               # Set guarantees of c2
c2.checkSat()                           # Saturate c2
c2.printInfo()                          # Print information of c2
 
c2.checkCompat(print_sol=True)          # Check compatibility of c2
c2.checkConsis(print_sol=True)          # Check consistency of c2
c2.checkFeas(print_sol=True)            # Check feasiblity of c2

c3 = contract('c3')                     # Create a contract c3
c3.set_deter_uncontrolled_vars(['x'])   # Set a deterministic uncontrolled variable
c3.set_controlled_vars(['y'])           # Set a controlled variable

c3.set_assume('(1 <= x)')               # Set assumptions of c3
c3.set_guaran('(1 <= y)')               # Set guarantees of c3
c3.checkSat()                           # Saturate c3
c3.printInfo()                          # Print information of c3

c3.checkCompat(print_sol=True)          # Check compatibility of c3
c3.checkConsis(print_sol=True)          # Check consistency of c3
c3.checkFeas(print_sol=True)            # Check feasiblity of c3

c4 = contract('c4')                     # Create a contract c4
c4.set_deter_uncontrolled_vars(['x'])   # Set a deterministic uncontrolled variable
c4.set_controlled_vars(['y', 'z'])      # Set a controlled variable

c4.set_assume('(4 <= x)')               # Set assumptions of c4
c4.set_guaran('((4 <= y) & (z <= -1))') # Set guarantees of c4
c4.checkSat()                           # Saturate c4
c4.printInfo()                          # Print information of c4

c4.checkCompat(print_sol=True)          # Check compatibility of c4
c4.checkConsis(print_sol=True)          # Check consistency of c4
c4.checkFeas(print_sol=True)            # Check feasiblity of c4

c2.checkRefine(c1, print_sol=True)      # Check whether c2 refines c1
c3.checkRefine(c1, print_sol=True)      # Check whether c3 refines c1
c4.checkRefine(c1, print_sol=True)      # Check whether c4 refines c1
