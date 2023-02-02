from typing import Union, List, Tuple, Optional
import numpy as np
from collections import deque

from highway.road.lane import AbstractLane
from highway.road.road import Road, LaneIndex
from highway.vehicle.road_object import RoadObject
from highway.util import wrap_to_pi


class Vehicle(RoadObject):

    """
    A moving vehicle on a road, and its kinematics.

    The vehicle is represented by a dynamical system: a modified bicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """

    DEFAULT_LENGTH = 3.0
    """ Vehicle length [m] """
    DEFAULT_WIDTH = 2.0
    """ Vehicle width [m] """
    DEFAULT_SPEEDS = [23, 25]
    """ Range for random initial speeds [m/s] """

    def __init__(self,
                 env: 'AbstractEnv',
                 position: np.ndarray,
                 heading: float = 0,
                 speed: float = 0,
                 follow: bool = False,
                 color: Optional[Tuple[float, float, float]] = None):
        super().__init__(env, position, heading, speed, color)
        self.action = np.array([0.0, 0.0])
        self.fuel_consumption = 0
        self.crashed = False

        self.follow = follow
        self.follow_speed = speed if follow else -1
        self.follow_lane_index = self.lane_index
        self.follow_lane = self.lane

        self.route = []
        self.target = np.empty(2)

        self._observe()

    @classmethod
    def create_random(cls, env: 'AbstractEnv',
                      lane_from: Optional[str] = None,
                      lane_to: Optional[str] = None,
                      lane_id: Optional[int] = None,
                      speed: float = None,
                      spacing: float = 1,
                      follow: bool = False,
                      color: Optional[Tuple[float, float, float]] = None) -> "Vehicle":
        """
        Create a random vehicle on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param env: the environment
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or speed
        """
        road = env.road

        _from = lane_from or env.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or env.np_random.choice(list(road.network.graph[_from].keys()))
        _id = lane_id if lane_id is not None else env.np_random.choice(len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = env.np_random.uniform(0.7*lane.speed_limit, 0.8*lane.speed_limit)
            else:
                speed = env.np_random.uniform(Vehicle.DEFAULT_SPEEDS[0], Vehicle.DEFAULT_SPEEDS[1])

        default_spacing = 2 * cls.DEFAULT_LENGTH + speed
        offset = spacing * default_spacing

        vehicle_on_lane = [v for v in env.controlled_vehicle + env.uncontrolled_vehicle if lane.on_lane_by_position(v.position)]
        x0 = np.max([lane.coordinate(v.position)[0] for v in vehicle_on_lane]) if vehicle_on_lane else 0
        x0 += offset
        v = cls(env, lane.position(x0, 0), lane.heading_at(x0), speed, follow, color)
        return v

    def plan_route(self, destination: str, lane_id : Optional[int] = None, offset = 50) -> "ControlledVehicle":
        """
        Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        path = self.env.road.network.shortest_path(self.lane_index[0], destination)
        self.route = []
        for i in range(len(path) - 1):
            self.route.append([(path[i], path[i + 1], j) for j in range(len(self.env.road.network.graph[path[i]][path[i+1]]))])

        #  set target
        end_lane_id = lane_id if lane_id is not None else self.env.np_random.choice(len(self.route[-1]))
        end_lane = self.env.road.network.get_lane(self.route[-1][end_lane_id])

        vehicle_on_lane = [v for v in self.env.controlled_vehicle + self.env.uncontrolled_vehicle if end_lane.on_lane_by_position(v.target)]
        x0 = np.min([end_lane.coordinate(v.target)[0] for v in vehicle_on_lane]) - offset if vehicle_on_lane else end_lane.length
        self.target = end_lane.position(x0, 0)
        return self

    def act(self, action: np.ndarray = None) -> None:
        """
        Store an action to be repeated.

        :param action: the input action
        """
        if action is not None:
            self.action = action

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        if self.follow:
            self.follow_action()
        self.clip_action()
        #  print(self.action)
        #  print(self.velocity)
        #  print(dt)
        #  input()
        self.fuel_consumption += np.sum(np.abs(self.action) * dt)

        self.position += self.velocity * dt

        self.velocity += self.action * dt
        self.velocity = np.clip(self.velocity, -self.lane.speed_limit, self.lane.speed_limit)

        self.on_state_update()

    def clip_action(self) -> None:
        self.action.astype(float)

        acceleration_limit = self.env.config["road"]["acceleration_limit"]
        self.action = np.clip(self.action, -acceleration_limit, acceleration_limit)

        if self.crashed:
            self.action = self.velocity * -1.0

    def follow_action(self) -> None:
        curr_longitudinal, curr_lateral = self.follow_lane.coordinate(self.position)
        next_longitudinal = curr_longitudinal + self.speed * 0.1
        next_heading = self.follow_lane.heading_at(next_longitudinal)

        # Lateral speed to heading
        lateral_heading = np.arcsin(np.clip(- curr_lateral / (0.6 * self.speed), -1, 1))
        heading = wrap_to_pi(next_heading + np.clip(lateral_heading, -np.pi/4, np.pi/4) - self.heading)
        heading_command = np.arcsin(np.clip(5 * heading, -1, 1))

        # Heading control
        a = self.follow_speed * np.array([np.cos(heading_command + self.heading), np.sin(heading_command + self.heading)]) - self.speed * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.action = a
        #  print(self.follow_lane_index)
        #  print("longitudina: {}".format(longitudinal))
        #  print("lateral: {}".format(lateral))
        #  print("next_heading: {}".format(next_heading))
        #  print("curr_lateral: {}".format(curr_lateral))
        #  print("speed: {}".format(self.speed))
        #  print("lateral_heading: {}".format(lateral_heading))
        #  print("heading: {}".format(heading))
        #  print("heading_command: {}".format(heading_command))
        #  print("supposed velocity: {}".format(v))
        #  input()

    def on_state_update(self) -> None:
        if self.follow and self.lane.after_end_by_position(self.position):
            next_lane = self.env.road.next_lane(self.lane_index, self.position)
            self.follow_lane_index = next_lane if next_lane is not None else self.lane_index
            self.follow_lane = self.env.road.network.get_lane(self.follow_lane_index)
        self.lane_index = self.env.road.network.get_closest_lane_index(self.position, self.heading)
        self.lane = self.env.road.network.get_lane(self.lane_index)

        self._observe()

    def _observe(self) -> None:
        obs = self.env.observation.observe(self)
        reward = self._reward()
        terminal = self.is_terminal()
        self.info = {"observation": obs, "reward": reward, "is_terminal": terminal}

    def _reward(self) -> float:
        """
        Return the reward associated the current state.

        :return: the reward
        """
        if self.route:
            dis_target = np.linalg.norm(self.position - self.target)
        else:
            dis_target = 0
        on_road = int(self.lane.on_lane_by_position(self.position))
        crashed = int(self.crashed)

        return dis_target + on_road + crashed

    def is_terminal(self) -> bool:
        """
        Check whether the current state is a terminal state

        :return:is the state terminal
        """
        return self.crashed

    def region(self, extra_width = 0.5) -> List[List[np.ndarray]]:
        res = []
        for r in self.route:
            for index, lane_id in enumerate(r):
                lane = self.env.road.network.get_lane(lane_id)

                left_margin = self.DEFAULT_WIDTH / 2 + extra_width if index == 0 else 0
                right_margin = self.DEFAULT_WIDTH / 2 + extra_width if index == len(r) - 1 else 0

                res.append(lane.region(left_margin, right_margin))
        res = np.array(res)
        return res


