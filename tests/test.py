
import sys
from pystl import *
import numpy
import time


# Car
# Mass is a constant. I use the Chevy Bolt: 1616 Kg
# Maximum torque, 360 Nm
# Maximum braking force, 2000 N  
#
# Forces are taken from:
# f0 = 51    N
# f1 = 1.2567 Ns/m
# f2 = 0.4342 Ns^2/m^2
car = contract('car')

car.add_deter_vars(
    ['acceleration', 'f_w', 'f_r', 'f_e', 'f_b',
    'velocity',
    'position',
    'leading_velocity']
)

car.add_nondeter_vars(
    ['noise_velocity', 'noise_position', 'noise_leading_velocity'], 
    mean = [0, 0, 0], cov = [[0.00001, 0, 0], [0, 0.00002, 0], [0, 0, 0.00003]])

car.add_deter_vars(['sensed_velocity'])

car.set_assume(
       '(G[0,10] (position >= position - velocity)) \
           & (G[0,10] (leading_velocity >= 0))')

car.set_guaran(
    '(G[0,10] (P[0.999] (noise_velocity <= 0.01*velocity))) & \
    (G[0,10] (P[0.999] (noise_velocity >= -0.01*velocity))) & \
    (G[0,10] (P[0.999] (noise_position <= 0.01*position))) & \
    (G[0,10] (P[0.999] (noise_position >= -0.01*position))) & \
    (G[0,10] (P[0.999] (noise_leading_velocity <= 0.01*leading_velocity))) & \
    (G[0,10] (P[0.999] (noise_leading_velocity >= -0.01*leading_velocity))) & \
    (G[0,10] (1616*acceleration >= f_w - f_r)) & \
    (G[0,10] (f_w <= f_e - f_b)) & \
    (G[0,10] (f_r <= 51 + 1.2567*velocity + 0.4342*velocity^2)) & \
    (G[0,10](velocity <= velocity + acceleration)) & \
    (G[0,10](position >= position + leading_velocity - velocity))'
)

######### Braking system
brakes = contract('brakes')
brakes.add_nondeter_vars(['noise_f_b'], mean = [0], cov = [[0.00004]])

brakes.add_deter_vars(
    ['f_b', 'f_b_target']
)

brakes.set_assume(
    'G[0,10] ((f_b_target >= 0) & (f_b_target <= 2000))'
)

brakes.set_guaran(
    '(G[0,10] (P[0.999] (noise_f_b <= 0.01*f_b))) & \
        (G[0,10] (P[0.999] (noise_f_b >= -0.01*f_b))) & \
        (G[2,10] ((f_b_target - f_b <= 1) & (f_b_target - f_b >= -1))) & \
        (G[0,10] ((f_b >= 0) & (f_b <= 2000)))'
)

######### Engine
engine = contract('engine')
engine.add_nondeter_vars(['noise_f_e'], mean = [0], cov = [[0.00005]])

engine.add_deter_vars(
    ['f_e', 'f_e_target']
)

engine.set_assume(
    'G[0,10] ((f_e_target >= -360) & (f_e_target <= 360))'
)

engine.set_guaran(
    '(G[0,10] (P[0.999] (noise_f_e <= 0.01*f_e))) & \
        (G[0,10] (P[0.999] (noise_f_e >= -0.01*f_e))) & \
        (G[2,10] ((f_e_target - f_e <= 1) & (f_e_target - f_e >= -1))) & \
        (G[0,10] ((f_e >= -360) & (f_e <= 360)))'
)


########## Sensors

sensors = contract('sensors')
sensors.add_nondeter_vars(
    ['noise_velocity', 'noise_position', 'noise_leading_velocity', 
    'noise_f_b', 'noise_f_e'], 
    mean = [0, 0, 0, 0, 0], 
    cov = [[0.00001, 0, 0, 0, 0], [0, 0.00002, 0, 0, 0], 
    [0, 0, 0.00003, 0, 0], [0, 0, 0, 0.00004, 0], [0, 0, 0, 0, 0.00005]])

sensors.add_deter_vars(['velocity', 'sensed_velocity', 
    'position', 'sensed_position',
    'leading_velocity', 'sensed_leading_velocity',
    'sensed_f_e', 'sensed_f_b', 'f_e', 'f_b'])

sensors.set_assume(
    '(G[0,10] (P[0.999] (noise_velocity <= 0.01*velocity))) & \
    (G[0,10] (P[0.999] (noise_velocity >= -0.01*velocity))) & \
    (G[0,10] (P[0.999] (noise_position <= 0.01*position))) & \
    (G[0,10] (P[0.999] (noise_position >= -0.01*position))) & \
    (G[0,10] (P[0.999] (noise_leading_velocity <= 0.01*leading_velocity))) & \
    (G[0,10] (P[0.999] (noise_leading_velocity >= -0.01*leading_velocity))) & \
    (G[0,10] (P[0.999] (noise_f_e <= 0.01*f_e))) & \
    (G[0,10] (P[0.999] (noise_f_e >= -0.01*f_e))) & \
    (G[0,10] (P[0.999] (noise_f_b <= 0.01*f_b))) & \
    (G[0,10] (P[0.999] (noise_f_b >= -0.01*f_b)))')

sensors.set_guaran('(G[0,10] (sensed_velocity - velocity <= 0.2*velocity)) & \
    (G[0,10] (sensed_velocity - velocity >= -0.2*velocity)) & \
    (G[0,10] (sensed_position - position <= 0.02*position)) & \
    (G[0,10] (sensed_position - position >= -0.02*position)) & \
    (G[0,10] (sensed_leading_velocity - leading_velocity <= 0.2*leading_velocity)) & \
    (G[0,10] (sensed_leading_velocity - leading_velocity >= -0.2*leading_velocity)) & \
    (G[0,10] (sensed_f_e <= 0.02*f_e)) & \
    (G[0,10] (sensed_f_e >= -0.02*f_e)) & \
    (G[0,10] (sensed_f_b <= 0.02*f_b)) & \
    (G[0,10] (sensed_f_b >= -0.02*f_b))')

########## Controller

controller = contract('controller')
controller.add_deter_vars(['sensed_velocity', 'velocity',
    'position', 'sensed_position',
    'leading_velocity', 'sensed_leading_velocity',
    'sensed_f_b', 'sensed_f_e', 'f_b_target', 'f_e_target'])

controller.set_assume('(G[0,10] ((sensed_velocity - velocity <= 0.02*velocity) & \
    (sensed_velocity - velocity >= -0.2*velocity))) & \
        (G[0,10] ((sensed_position - position <= 0.02*position) & \
    (sensed_position - position >= -0.02*position))) & \
        (G[0,10] ((sensed_leading_velocity - leading_velocity <= 0.2*leading_velocity) & \
    (sensed_leading_velocity - leading_velocity >= -0.2*leading_velocity))) & \
        (G[0,10] (sensed_f_e < 2)) & \
        (G[0,10] (sensed_f_e > -2)) & \
        (G[0,10] (sensed_f_b < 2)) & \
        (G[0,10] (sensed_f_b > -2))')

controller.set_guaran(
    '(G[2,10] ((sensed_velocity >= 0) & (sensed_velocity <= 14))) & \
    (G[2,10] (sensed_position >= 10)) & \
    (G[0,10] (!((f_e_target > 0) & (f_b_target < 0)))) & \
    (G[0,10] ((f_e_target + f_b_target >= -2360) & (f_e_target <= 360)))')

controller.checkSat()
controller.printInfo()
sensors.checkSat()
sensors.printInfo()
brakes.checkSat()
brakes.printInfo()
engine.checkSat()
engine.printInfo()
car.checkSat()
car.printInfo()

car.checkFeas()
brakes.checkFeas()
engine.checkFeas()
controller.checkFeas()
sensors.checkFeas()

be = composition(brakes, engine)
be.checkSat()
be.printInfo()
be.checkFeas()

cc = composition(controller, car)
cc.checkSat()
cc.printInfo()
cc.checkFeas() # Does not work.

becc = composition(be, cc)
becc.checkSat()
becc.printInfo()
becc.checkFeas() # Does not work.

acc = composition(becc, sensors)
acc.checkSat()
acc.checkFeas() # Does not work.