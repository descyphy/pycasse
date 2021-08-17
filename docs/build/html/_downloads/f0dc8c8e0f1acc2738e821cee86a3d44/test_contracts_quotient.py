from pystl import *
import numpy as np

c = contract('c')                     # Create a contract c
c.set_assume('G[0,3](x>=5)')          # Set an assumption
c.set_guaran('G[1,4](y>=3)')          # Set a guarantee
c.set_deter_uncontrolled_vars(['x'])  # Set a deterministic uncontrolled variable
c.set_controlled_vars(['y'])          # Set a controlled variable
c.saturate()                          # Saturate c
c.printInfo()                         # Print c

c.checkCompat(print_sol=True)         # Check compatibility of c
c.checkConsis(print_sol=True)         # Check consistency of c
c.checkFeas(print_sol=True)           # Check feasiblity of c

c2 = contract('c2')                   # Create a contract c2
c2.set_assume('TRUE')                 # Set an assumption
c2.set_guaran('G[0,4](y>=2)')         # Set a guarantee
c2.set_deter_uncontrolled_vars(['x']) # Set a deterministic uncontrolled variable
c2.set_controlled_vars(['y'])         # Set a controlled variable
c2.saturate()                         # Saturate c2
c2.printInfo()                        # Print c2

c2.checkCompat(print_sol=True)        # Check compatibility of c2
c2.checkConsis(print_sol=True)        # Check consistency of c2
c2.checkFeas(print_sol=True)          # Check feasiblity of c2

c2_quo = quotient(c, c2)              # Quotient c/c2
c2_quo.saturate()                     # Saturate c2_quo
c2_quo.printInfo()                    # Print c2_quo

c2_quo.checkCompat(print_sol=True)    # Check compatibility of c2_quo
c2_quo.checkConsis(print_sol=True)    # Check consistency of c2_quo
c2_quo.checkFeas(print_sol=True)      # Check feasiblity of c2_quo

c2_comp = composition(c2_quo, c2)     # Composition of c2_quo and c2
c2_comp.saturate()                    # Saturate c2_comp
c2_comp.printInfo()                   # Print c2_comp

c2_comp.checkCompat(print_sol=True)   # Check compatibility of c2_comp
c2_comp.checkConsis(print_sol=True)   # Check consistency of c2_comp
c2_comp.checkFeas(print_sol=True)     # Check feasiblity of c2_comp

c.checkRefine(c2_comp)                # Check whether c2_comp refines c