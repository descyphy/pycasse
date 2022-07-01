from pystl import *
import numpy as np

c = contract('c')                     # Create a contract c
c.add_deter_vars(['x', 'y'])          # Set a controlled variable
c.set_assume('G[0,3] (x => 5)')     # Set/define the assumptions
c.set_guaran('G[1,4] (y => 3)')     # Set/define the guarantees
c.checkSat()                          # Saturate c
c.printInfo()                         # Print c

c.checkCompat(print_sol=True)         # Check compatibility of c
c.checkConsis(print_sol=True)         # Check consistency of c
c.checkFeas(print_sol=True)           # Check feasiblity of c

c2 = contract('c2')                   # Create a contract c2
c2.add_deter_vars(['x', 'y'])          # Set a controlled variable
c2.set_assume('True')                 # Set/define the assumptions
c2.set_guaran('G[0,4] (5 <= y)')    # Set/define the guarantees
c2.checkSat()                         # Saturate c2
c2.printInfo()                        # Print c2

c2.checkCompat(print_sol=True)        # Check compatibility of c2
c2.checkConsis(print_sol=True)        # Check consistency of c2
c2.checkFeas(print_sol=True)          # Check feasiblity of c2

c2_quo = quotient(c, c2)              # Quotient c/c2
c2_quo.checkSat()                     # Saturate c2_quo
c2_quo.printInfo()                    # Print c2_quo

c2_quo.checkCompat(print_sol=True)    # Check compatibility of c2_quo
c2_quo.checkConsis(print_sol=True)    # Check consistency of c2_quo
c2_quo.checkFeas(print_sol=True)      # Check feasiblity of c2_quo

c2_comp = composition([c2_quo, c2])   # Composition of c2_quo and c2
c2_comp.checkSat()                    # Saturate c2_comp
c2_comp.printInfo()                   # Print c2_comp

c2_comp.checkCompat(print_sol=True)   # Check compatibility of c2_comp
c2_comp.checkConsis(print_sol=True)   # Check consistency of c2_comp
c2_comp.checkFeas(print_sol=True)     # Check feasiblity of c2_comp

c2_comp.checkRefine(c)                # Check whether c2_comp refines c
