import sys
sys.path.append("/home/kevin/Github/pystl")
from pystl import *
import numpy as np

c1 = contract('c1')                   # Create a contract c1
c1.set_deter_uncontrolled_vars(['x']) # Set a deterministic uncontrolled variable
c1.set_controlled_vars(['y'])         # Set a controlled variable
c1.set_assume('(G[0,3] (x => 5))')    # Set/define the assumptions
c1.set_guaran('(F[1,4] (y => 1))')    # Set/define the guarantees
c1.checkSat()                         # Saturate c1
c1.printInfo()                        # Print c1

#  c1.checkCompat(print_sol=True)        # Check compatibility of c1
#  c1.checkConsis(print_sol=True)        # Check consistency of c1
#  c1.checkFeas(print_sol=True)          # Check feasiblity of c1

c2 = contract('c2')                   # Create a contract c2
c2.set_controlled_vars(['y'])         # Set a controlled variable
c2.set_controlled_vars(['z'])         # Set a controlled variable
c2.set_assume('True')                 # Set/define the assumptions
c2.set_guaran('(G[0,4] (y <= 2))')    # Set/define the guarantees
c2.checkSat()                         # Saturate c2
c2.printInfo()                        # Print c2

#  c2.checkCompat(print_sol=True)        # Check compatibility of c2
#  c2.checkConsis(print_sol=True)        # Check consistency of c2
#  c2.checkFeas(print_sol=True)          # Check feasiblity of c2

#  c12_conj = conjunction(c1, c2)        # Conjunction of c1 and c2
#  c12_conj.checkSat()                   # Saturate c12_conj
#  c12_conj.printInfo()                  # Print c12_conj
#
#  c12_conj.checkCompat(print_sol=True)  # Check compatibility of c12_conj
#  c12_conj.checkConsis(print_sol=True)  # Check consistency of c12_conj
#  c12_conj.checkFeas(print_sol=True)    # Check feasiblity of c12_conj

c12_comp = composition(c1, c2)        # Composition of c1 and c2
c12_comp.checkSat()                   # Saturate c12_comp
c12_comp.printInfo()                  # Print c12_comp

#  c12_comp.checkCompat(print_sol=True)  # Check compatibility of c12_comp
#  c12_comp.checkConsis(print_sol=True)  # Check consistency of c12_comp
#  c12_comp.checkFeas(print_sol=True)    # Check feasiblity of c12_comp
