from copy import copy, deepcopy
import gym
from gym.utils import seeding
import numpy as np
from typing import Tuple, List

from highway_custom.observation import Observer
from highway_custom.graphic import Graphic
from highway_custom.road.lane import StraightLane, CircularLane
from highway_custom.util import write_json, read_json

import time, os

EPS = float(1)

class AbstractEnv(gym.Env):
    """
    A generic environment for various tasks involving a vehicle driving on a road.

    The environment contains a road populated with vehicles, and multiple controlled ego-vehicles.
    """

    PERCEPTION_DISTANCE = 180.0

    __slots__ = ('config', "start_time", "np_random", "num_policy", "num_simulation", "road", "controlled_vehicle", "uncontrolled_vehicle", "observation")

    def __init__(self, config: dict = None) -> None:
        # Configuration
        self.config = self.default_config()
        self.configure(config)

        # Seeding
        self.start_time = time.time()
        self.np_random = None
        self.seed()

        # Running
        self.num_policy = 0  # Actions performed
        self.num_simulation = 0  # Simulation time

        # Scene
        self.road = None
        self.controlled_vehicle = []
        self.uncontrolled_vehicle = []

        # Observation
        self.observation = Observer(self, **self.config["observation"])

        self.reset()

    @staticmethod
    def default_config() -> dict:
        """
        Default environment configuration.

        Can be overloaded in environment implementations, or by calling configure().
        :return: a configuration dict
        """
        return {
            "simulation_policy_frequency_ratio": 1,  # [Hz]
            "simulation_frequency": 15,  # [Hz]
            "observation": {},
            "road": {},
            "vehicle": {},
            "graphic": {
                "screen_width": 600,  # [px]
                "screen_height": 150,  # [px]
                "screen_centering_position": [0.3, 0.5],
                "road_centering_position": [100, 100],
                "scaling": 5.5,
                "show_trajectory": False,
                "trajectory_frequency": 2,
                }
        }

    def configure(self, config: dict) -> None:
        if config:
            for k, v in config.items():
                if k in ("observation", "road", "vehicle", "graphic"):
                    self.config[k].update(v)
                else:
                    self.config[k] = v

    def write_config(self, file_path) -> None:
        write_json(self.config, file_path)

    def read_config(self, file_path) -> None:
        self.configure(read_json(file_path))

    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> np.ndarray:
        """
        Reset the environment to it's initial configuration

        :return: the observation of the reset state
        """
        self.num_policy = self.num_simulation = 0
        self._reset()

    def _reset(self) -> None:
        """
        Reset the scene: roads and vehicles.

        This method must be overloaded by the environments.
        """
        raise NotImplementedError()

    def step(self, action) -> None:
        """
        Step the environment dynamics.

        All vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.
        """
        assert(not action)

        if self.road is None or self.controlled_vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")
        else:
            self.num_policy += 1
            self._simulate()
        
        return None, None, None, None

    def _simulate(self) -> None:
        """Perform several steps of simulation with constant action."""
        for _ in range(int(self.config["simulation_policy_frequency_ratio"])):
            self.road.step(1 / self.config["simulation_frequency"])
            self.num_simulation += 1

    def is_terminal(self) -> bool:
        """
        Check whether the current state is a terminal state

        :return:is the state terminal
        """
        vehicle_crash = [v.is_terminal() for v in self.controlled_vehicle]
        vehicle_stop = [np.sum(np.abs(v.position - v.target)) <= EPS for v in self.controlled_vehicle]
        #  print(vehicle_crash)
        #  print(vehicle_stop)
        return any(vehicle_crash) or all(vehicle_stop)

    def state(self) -> None:
        res = []
        for v in self.controlled_vehicle:
            res.append([*v.position, *v.velocity])
        return np.array(res)

    def print_state(self, time) -> None:
        print("Vehicle States at time {}: ".format(time))
        for index, vehicle in enumerate(self.controlled_vehicle):
            print("   vehicle id: {}:".format(index))
            print("      position: {}, {}".format(*vehicle.position))
            print("      velocity: {}, {}".format(*vehicle.velocity))
        print()

    def print_target(self) -> None:
        print("Vehicle Goal:")
        for index, vehicle in enumerate(self.controlled_vehicle):
            print("   vehicle id: {}:".format(index))
            print("      target: {}, {}".format(*vehicle.target))
        print()

    def print_result(self, elapsed_time) -> None:
        print("Total elapsed time: {}".format(elapsed_time))
        print("Total synthesis time: {}".format(time.time() - self.start_time))
        print("Vehicle States:")
        total_distance = 0
        total_fuel_consumption = 0
        for index, vehicle in enumerate(self.controlled_vehicle):
            print("   vehicle id: {}:".format(index))
            print("      position: {}, {}".format(*vehicle.position))
            print("      velocity: {}, {}".format(*vehicle.velocity))
            print("      distance: {}".format(np.linalg.norm(vehicle.position - vehicle.target)))
            print("      fuel consumption: {}".format(vehicle.fuel_consumption))
            total_distance += np.linalg.norm(vehicle.position - vehicle.target)
            total_fuel_consumption += vehicle.fuel_consumption
        print("Total distance: {}".format(total_distance))
        print("Total fuel consumption: {}".format(total_fuel_consumption))
        print()

    def save_result(self, elapsed_time, file_path) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write("Total elapsed time: {}\n".format(elapsed_time))
            f.write("Total synthesis time: {}\n".format(time.time() - self.start_time))
            f.write("Vehicle States:\n")
            total_distance = 0
            total_fuel_consumption = 0
            for index, vehicle in enumerate(self.controlled_vehicle):
                f.write("   vehicle id: {}:\n".format(index))
                f.write("      position: {}, {}\n".format(*vehicle.position))
                f.write("      velocity: {}, {}\n".format(*vehicle.velocity))
                f.write("      distance: {}\n".format(np.linalg.norm(vehicle.position - vehicle.target)))
                f.write("      fuel consumption: {}\n".format(vehicle.fuel_consumption))
                total_distance += np.linalg.norm(vehicle.position - vehicle.target)
                total_fuel_consumption += vehicle.fuel_consumption
            f.write("Total distance: {}\n".format(total_distance))
            f.write("Total fuel consumption: {}\n".format(total_fuel_consumption))
