import gym
import numpy as np
import os, sys

import highway_custom
from highway_custom.graphic import Graphic
from pycasse import Controller

DEBUG = False
SIMU_TIME = 20
SIMU_FREQUENCY = 4
H = 4
VEHICLE_NUM = 3
SCENARIO = "cooperating"
# SCENARIO = "noncooperating"
if SCENARIO == "noncooperating":
    GROUP_NUM = VEHICLE_NUM
elif SCENARIO == "cooperating":
    GROUP_NUM = 1
else:
    assert(False)

env = gym.make('intersection-v1')

env.configure({
    "simulation_frequency": SIMU_FREQUENCY,
    "vehicle": {
        "controlled_vehicle": VEHICLE_NUM,
        "controlled_speed": 5,
        "controlled_spacing": [1.2, 1, 1],
        "controlled_target": [2, 3, 1],
        "uncontrolled_vehicle": 0
        },
    "road": {
        "lane_length": 50,
        "extra_width": 10,
        "num_way": 4,
        },
    "graphic": {
        "show_trajectory": True,
        "trajectory_frequency": 1,
        }
})
env.reset()

viewer = Graphic(env)

# Print out the goal states
env.print_state(0)
env.print_target()

# Loading the environment to PySTL
pystl_controller = Controller(env, H * SIMU_FREQUENCY, debug = DEBUG)

# Run Simulation
t = 0
while not env.is_terminal() and t < SIMU_TIME * SIMU_FREQUENCY:
    viewer.record()
    viewer.display()

    # Find the current states of the vehicles
    state = env.state()

    # Print out the states
    if DEBUG:
        #  env.print_state(t/SIMU_FREQUENCY)
        env.print_state(t)

    # Find the control trajectory using PySTL
    #  synthesis_fail = pystl_controller.optimize_model(state)
    synthesis_fail = pystl_controller.optimize_model(state, GROUP_NUM)
    control_input = pystl_controller.find_control(0)
    if synthesis_fail:
        print("Synthesis failure.")
        break
    print("Control at time {}: {}".format(t/SIMU_FREQUENCY, control_input))

    # Apply control to vehicles
    for vehicle_num in range(VEHICLE_NUM):
        env.controlled_vehicle[vehicle_num].act(np.array(control_input[vehicle_num]))
    
    env.step([])
    t += 1

    if DEBUG:
        input()

# View and save the results
env.print_result(t/ SIMU_FREQUENCY)
env.save_result(t/ SIMU_FREQUENCY, "{}{}record{}intersection_{}{}.txt".format(os.getcwd(), os.sep, os.sep, SCENARIO, VEHICLE_NUM))
viewer.display(file_path="{}{}record{}intersection_{}{}.mp4".format(os.getcwd(), os.sep, os.sep, SCENARIO, VEHICLE_NUM), from_start = True)

# Close the environment
env.close()

