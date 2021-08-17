from pystl import *
import numpy as np

c1 = contract('c1')                   # Create a contract c1
c1.set_deter_uncontrolled_vars(['x']) # Set a deterministic uncontrolled variable
c1.set_controlled_vars(['y'])         # Set a controlled variable

c1.set_assume('(G[0,3] (x => 5))')    # Set assumptions of c1
c1.set_guaran('(G[1,3] (y => 2))')    # Set guarantees of c1
c1.checkSat()                         # Saturate c1
c1.printInfo()                        # Print information of c1

c1.checkCompat(print_sol=True)        # Check compatibility of c1
c1.checkConsis(print_sol=True)        # Check consistency of c1
c1.checkFeas(print_sol=True)          # Check feasiblity of c1

c2 = contract('c2')                   # Create a contract c2
c2.set_deter_uncontrolled_vars(['x']) # Set a deterministic uncontrolled variable
c2.set_controlled_vars(['y'])         # Set a controlled variable

c2.set_assume('(F[0,3] (x => 5))')    # Set assumptions of c2
c2.set_guaran('(G[0,3] (y => 3))')    # Set guarantees of c2
c2.checkSat()                         # Saturate c2
c2.printInfo()                        # Print information of c2

c2.checkCompat(print_sol=True)        # Check compatibility of c2
c2.checkConsis(print_sol=True)        # Check consistency of c2
c2.checkFeas(print_sol=True)          # Check feasiblity of c2

c3 = contract('c3')                   # Create a contract c3
c3.set_deter_uncontrolled_vars(['x']) # Set a deterministic uncontrolled variable
c3.set_controlled_vars(['y'])         # Set a controlled variable

c3.set_assume('(F[0,3] (x => 5))')    # Set assumptions of c3
c3.set_guaran('(F[1,3] (y => 3))')    # Set guarantees of c3
c3.checkSat()                         # Saturate c3
c3.printInfo()                        # Print information of c3

c3.checkCompat(print_sol=True)        # Check compatibility of c3
c3.checkConsis(print_sol=True)        # Check consistency of c3
c3.checkFeas(print_sol=True)          # Check feasiblity of c3

c2.checkRefine(c1, print_sol=True)    # Check whether c2 refines c1
c3.checkRefine(c1, print_sol=True)    # Check whether c3 refines c1
