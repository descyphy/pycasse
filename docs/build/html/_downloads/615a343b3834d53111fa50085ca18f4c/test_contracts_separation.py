from pystl import *
import numpy as np

c = contract('c')                       # Create a contract c
c.set_assume('G[0,3](5<=x)')            # Set an assumption
c.set_guaran('G[1,4](1<=y<=3)')         # Set a guarantee
c.set_deter_uncontrollable_vars(['x'])  # Set a deterministic uncontrollable variable
c.set_controllable_vars(['y'])          # Set a controllable variable
c.saturate()                            # Saturate c
c.printInfo()                           # Print c

c.checkCompat(print_sol=True)           # Check compatibility of c
c.checkConsis(print_sol=True)           # Check consistency of c
c.checkFeas(print_sol=True)             # Check feasiblity of c

c2 = contract('c2')                     # Create a contract c2
c2.set_assume('TRUE')                   # Set an assumption
c2.set_guaran('G[0,4](y<=2)')           # Set a guarantee
c2.set_deter_uncontrollable_vars(['x']) # Set a deterministic uncontrollable variable
c2.set_controllable_vars(['y'])         # Set a controllable variable
c2.saturate()                           # Saturate c2
c2.printInfo()                          # Print c2

c2.checkCompat(print_sol=True)          # Check compatibility of c2
c2.checkConsis(print_sol=True)          # Check consistency of c2
c2.checkFeas(print_sol=True)            # Check feasiblity of c2

c2_sep = separation(c, c2)              # Separation c%c2
c2_sep.saturate()                       # Saturate c2_sep
c2_sep.printInfo()                      # Print c2_sep

c2_sep.checkCompat(print_sol=True)      # Check compatibility of c2_sep
c2_sep.checkConsis(print_sol=True)      # Check consistency of c2_sep
c2_sep.checkFeas(print_sol=True)        # Check feasiblity of c2_sep

c2_merge = merge(c2_sep, c2)            # Merge of c2_sep and c2
c2_merge.saturate()                     # Saturate c2_merge
c2_merge.printInfo()                    # Print c2_merge

c2_merge.checkCompat(print_sol=True)    # Check compatibility of c2_merge
c2_merge.checkConsis(print_sol=True)    # Check consistency of c2_merge
c2_merge.checkFeas(print_sol=True)      # Check feasiblity of c2_merge

c2_merge.checkRefine(c)                 # Check whether c refines c2_merge