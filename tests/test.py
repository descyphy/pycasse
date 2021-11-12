import sys
from pystl import *
import numpy as np
import time
# Car
# Mass is a constant. I use the Chevy Bolt: 1616 Kg
# Maximum torque, 360 Nm
# Maximum braking force, 2000 N  
#
# Forces are taken from:
# f0 = 51 N
# f1 = 1.2567 Ns/m
# f2 = 0.4342 Ns^2/m^2
car = contract('car')
car.add_deter_vars(
    ['acceleration', 'f_w', 'f_r', 'f_e', 'f_b',
    'velocity', 'sensed_velocity',
    'distance', 'sensed_distance',
    'leading_velocity'
    ])
car.set_assume(
       '(G[0,10] (distance >= sensed_distance - velocity)) \
           & (G[0,2] (leading_velocity >= 0)) \
               & (G[0,2] (leading_velocity >= 0))')
car.set_guaran(
    '(G[0,10] (1616*acceleration >= f_w - f_r)) & \
        (G[0,10] (f_w <= f_e + f_b)) & \
        (G[0,10] (f_r <= 51 + 1.2567*velocity + 0.4342*velocity^2)) & \
        (G[0,10](velocity <= sensed_velocity + acceleration)) & \
        (G[0,10](distance >= sensed_distance + leading_velocity - velocity))'
    )
car.checkCompat(print_sol=False)
car.checkConsis(print_sol=False)
car.checkFeas(print_sol=False)

# Braking system
brakes = contract('brakes')
brakes.add_deter_vars(
    ['f_b', 'f_b_sensed', 'f_b_target']
)
brakes.set_assume(
    '(G[2,10] ((f_b_sensed - f_b >= -1) & (f_b_sensed - f_b <= 1))) & \
        (G[2,10] ((f_b_target <= 0) & (f_b_target >= -2000)))'
)
brakes.set_guaran(
    'G[2,10] ((f_b_target - f_b <= 1) & (f_b_target - f_b <= -1))'
)

brakes.checkCompat(print_sol=False)
brakes.checkConsis(print_sol=False)
brakes.checkFeas(print_sol=False)

# engine
engine = contract('engine')
engine.add_deter_vars(
    ['f_e', 'f_e_sensed', 'f_e_target']
)
engine.set_assume(
    '(G[2,10] ((f_e_sensed - f_e >= -1) & (f_e_sensed - f_e <= 1))) & \
        (G[2,10] ((f_e_target >= -360) & (f_e_target <= 360)))'
)
engine.set_guaran(
    'G[2,10] ((f_e_target - f_e <= 1) & (f_e_target - f_e <= -1))'
)

engine.checkCompat(print_sol=False)
engine.checkConsis(print_sol=False)
engine.checkFeas(print_sol=False)

# controller
controller = contract('controller')
controller.add_deter_vars(
    ['sensed_velocity', 'velocity',
    'position', 'sensed_position',
    'f_e', 'f_b', 'f_e_target', 'f_b_target',
    'f_e_sensed', 'f_b_sensed'
    ]
)
controller.set_assume(
    '(G[0,10]((sensed_velocity - velocity <= 0.01*velocity) & \
        (sensed_velocity - velocity >= -0.01*velocity))) & \
    (G[0,10]((sensed_position - position <= 0.01*position) & \
        (sensed_position - position >= -0.01*position))) & \
    (G[2,10] ((f_e - f_e_sensed <= 1) & (f_e - f_e_sensed >= -1))) & \
    (G[2,10] ((f_b - f_b_sensed <= 1) & (f_b - f_b_sensed >= -1))) & \
    (G[2,10] (1616*velocity <= sensed_velocity + f_b_sensed + f_e_sensed)) & \
    (G[2,10] (position <= sensed_position - velocity))'
)
controller.set_guaran(
    '(G[2,10] ((sensed_velocity >= 0) & (sensed_velocity <= 14))) & \
    (G[2,10] (sensed_position >= 10)) & \
    (G[0,10] (!((f_e_target > 0) & (f_b_target < 0)))) & \
    (G[0,10] ((f_e_target + f_b_target >= -2360) & (f_e_target <= 360)))')

controller.checkCompat(print_sol=False)
controller.checkConsis(print_sol=False)
controller.checkFeas(print_sol=False)