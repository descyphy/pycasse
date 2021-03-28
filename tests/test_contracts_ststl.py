import sys
sys.path.append("/home/kevin/Github/pystl")
from pystl import *
import numpy as np

c1 = contract('c1')                                  # Create a contract c1
c1.set_assume('(G[0,3] (<= 5 x))')                   # Set/define the assumptions
c1.set_guaran('(G[1,3] (P[0.85] (<= y-2w1+3w2 8)))') # Set/define the guarantees
c1.set_deter_uncontrolled_vars(['x'])                # Set a deterministic uncontrolled variable
c1.set_nondeter_uncontrolled_vars(['w1', 'w2'],  mean = np.array([[0], [2]]), \
                               cov = np.array([[1**2, 0], [0, 1**2]]), \
                               dtype='GAUSSIAN')     # Set nondeterministic uncontrolled variables
c1.set_controlled_vars(['y'])                        # Set a controlled variable
c1.saturate()                                        # Saturate c1
c1.printInfo()                                       # Print c1

c1.checkCompat(print_sol=True)                       # Check compatibility of c1
c1.checkConsis(print_sol=True)                       # Check consistency of c1
c1.checkFeas(print_sol=True)                         # Check feasibility of c1

c2 = contract('c2')                                  # Create a contract c2
c2.set_assume('(F[1,2] (<= 4 x))')                   # Set/define the assumptions
c2.set_guaran('(G[1,3] (P[0.95] (<= y-2w1+3w2 8)))') # Set/define the guarantees such that c2 refines c1
c2.set_deter_uncontrolled_vars(['x'])                # Set a deterministic uncontrolled variable
c2.set_nondeter_uncontrolled_vars(['w1', 'w2'],  mean = np.array([[0], [2]]), \
                               cov = np.array([[1**2, 0], [0, 1**2]]), \
                               dtype='GAUSSIAN')     # Set nondeterministic uncontrolled variables
c2.set_controlled_vars(['y'])                        # Set a controlled variable
c2.saturate()                                        # Saturate c2
c2.printInfo()                                       # Print c2

c2.checkCompat()                                     # Check compatibility of c2
c2.checkConsis()                                     # Check consistency of c2
c2.checkFeas()                                       # Check feasiblity of c2

c3 = contract('c3')                                  # Create a contract c3
c3.set_assume('(F[1,2] (<= 4 x))')                   # Set/define the assumptions
c3.set_guaran('(G[1,3] (P[0.75] (<= y-2w1+3w2 8)))') # Set/define the guarantees such that c3 does not refines c1
c3.set_deter_uncontrolled_vars(['x'])                # Set a deterministic uncontrolled variable
c3.set_nondeter_uncontrolled_vars(['w1', 'w2'],  mean = np.array([[0], [2]]), \
                               cov = np.array([[1**2, 0], [0, 1**2]]), \
                               dtype='GAUSSIAN')     # Set nondeterministic uncontrolled variables
c3.set_controlled_vars(['y'])                        # Set a controlled variable
c3.saturate()                                        # Saturate c3
c3.printInfo()                                       # Print c3

c3.checkCompat()                                     # Check compatibility of c3
c3.checkConsis()                                     # Check consistency of c3
c3.checkFeas()                                       # Check feasiblity of c3
   
c2.checkRefine(c1, print_sol=True)                   # Check whether c2 refines c1
c3.checkRefine(c1, print_sol=True)                   # Check whether c3 refines c1
