from pycasse import *

c1 = contract('c1')                                      # Create a contract c1
c1.add_deter_vars(['x', 'y'])                            # Set deterministic variables
c1.add_nondeter_vars(['w1', 'w2'], \
        mean = [0, 2], cov = [[1**2, 0], [0, 1**2]])     # Set nondeterministic variables
c1.set_assume('G[0,3] (5 <= x)')                         # Set assumptions of c1
c1.set_guaran('G[1,3] (P[0.95] (y - 2*w1 + 3*w2 <= 8))') # Set guarantees of c1
c1.checkSat()                                            # Saturate c1
c1.printInfo()                                           # Print c1

c1.checkCompat(print_sol=True)                           # Check compatibility of c1
c1.checkConsis(print_sol=True)                           # Check consistency of c1
c1.checkFeas(print_sol=True)                             # Check feasibility of c1

c2 = contract('c2')                                      # Create a contract c2
c2.add_deter_vars(['x', 'y'])                            # Set deterministic variables
c2.add_nondeter_vars(['w1', 'w2'], \
        mean = [0, 2], cov = [[1**2, 0], [0, 1**2]])     # Set nondeterministic variables
c2.set_assume('F[1,2] (4 <= x)')                         # Set/define the assumptions
c2.set_guaran('G[1,3] (P[0.97] (y - 2*w1 + 3*w2 <= 8))') # Set/define the guarantees such that c2 refines c1
c2.checkSat()                                            # Saturate c2
c2.printInfo()                                           # Print c2

c2.checkCompat(print_sol=True)                           # Check compatibility of c2
c2.checkConsis(print_sol=True)                           # Check consistency of c2
c2.checkFeas(print_sol=True)                             # Check feasiblity of c2

c3 = contract('c3')                                      # Create a contract c3
c3.add_deter_vars(['x', 'y'])                            # Set deterministic variables
c3.add_nondeter_vars(['w1', 'w2'], \
        mean = [0, 2], cov = [[1**2, 0], [0, 1**2]])     # Set nondeterministic variables
c3.set_assume('F[1,2] (4 <= x)')                         # Set/define the assumptions
c3.set_guaran('G[1,3] (P[0.65] (y - 2*w1 + 3*w2 <= 8))') # Set/define the guarantees such that c3 does not refines c1
c3.checkSat()                                            # Saturate c3
c3.printInfo()                                           # Print c3

c3.checkCompat(print_sol=True)                           # Check compatibility of c3
c3.checkConsis(print_sol=True)                           # Check consistency of c3
c3.checkFeas(print_sol=True)                             # Check feasiblity of c3

c2.checkRefine(c1, print_sol=True)                       # Check whether c2 refines c1
c3.checkRefine(c1, print_sol=True)                       # Check whether c3 refines c1
