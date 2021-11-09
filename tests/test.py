from pystl import *
# Car
# Mass is a constant. I use the Chevy Bolt: 1616 Kg
# Maximum torque, 430 Nm
car = contract('car')
car.add_deter_vars(['acceleration', 'f_w', 'f_r', 'velocity', 'distance', 'sensed_distance'])

car.set_assume('G[0,1] (distance >= sensed_distance - velocity)')
car.set_guaran('G[0,1] (1616*acceleration < f_w - f_r)')
