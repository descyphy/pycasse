import numpy as np
from gym.envs.registration import register

from highway.env.abstract_env import AbstractEnv
from highway.road.lane import LineType, AbstractLane, StraightLane, CircularLane
from highway.road.road import Road, RoadNetwork
from highway.vehicle.vehicle import Vehicle
from highway.graphic import VehicleGraphics


class MergeEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "road": {
                "speed_limit": 5,
                "acceleration_limit": 30
                },
            "vehicle": {
                "controlled_vehicle": 1,
                "controlled_start": [0],
                "controlled_target": [0],
                "controlled_speed": 5,
                "controlled_spacing": [2],
                "uncontrolled_vehicle": 1,
                "uncontrolled_spacing": 1,
                },
            "graphic": {
                "screen_width": 1500,
                "screen_height": 350,
                "screen_centering_position": [0.5, 0.5],
                "road_centering_position": [75, -2.5],
                "scaling": 9
                },
        })
        return config

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicle()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        ends = [50, 10, 30, 10, 50]  # Before, converging, merge, diverging, after
        lane_width = StraightLane.DEFAULT_LANE_WIDTH
        amplitude = 3.25

        # Highway lanes
        c, s, n = LineType.CONTINUOUS, LineType.STRIPED, LineType.NONE
        y = [lane_width, 0]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_type=line_type[i], speed_limit = self.config["road"]["speed_limit"]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_type=line_type_merge[i], speed_limit = self.config["road"]["speed_limit"]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_type=line_type[i], speed_limit = self.config["road"]["speed_limit"]))

        #  # Merging lane
        start = [0, - (amplitude * 2) - lane_width]
        end = [ends[0], - (amplitude * 2) - lane_width]
        ljk = StraightLane(start, end, line_type=[c, c], speed_limit = self.config["road"]["speed_limit"])
        net.add_lane("j", "k", ljk)

        start = [ends[0], - (amplitude * 2) - lane_width]
        end = [sum(ends[:2]), -lane_width]
        lkb = StraightLane(start, end, line_type=[c, c], speed_limit = self.config["road"]["speed_limit"])
        net.add_lane("k", "b", lkb)

        start = [sum(ends[:2]), - lane_width]
        end = [sum(ends[:3]), - lane_width]
        lbc = StraightLane(start, end, line_type=[n, c])
        net.add_lane("b", "c", lbc)

        start = [sum(ends[:3]), -lane_width]
        end = [sum(ends[:4]), - (amplitude * 2) - lane_width]
        lcm = StraightLane(start, end, line_type=[c, c], speed_limit = self.config["road"]["speed_limit"])
        net.add_lane("c", "m", lcm)

        start = [sum(ends[:4]), - (amplitude * 2) - lane_width]
        end = [sum(ends), - (amplitude * 2) - lane_width]
        lmn = StraightLane(start, end, line_type=[c, c], speed_limit = self.config["road"]["speed_limit"])
        net.add_lane("m", "n", lmn)

        road = Road(self, network=net)
        self.road = road

    def _make_vehicle(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        # Controlled vehicles
        self.controlled_vehicle = []
        self.uncontrolled_vehicle = []

        color = (VehicleGraphics.RED, VehicleGraphics.GREEN, VehicleGraphics.BLUE, VehicleGraphics.YELLOW, VehicleGraphics.PURPLE)
        for v_id in range(self.config["vehicle"]["controlled_vehicle"]):
            start_choice = self.config["vehicle"]["controlled_start"][v_id] if self.config["vehicle"]["controlled_start"] else self.np_random.choice(3)
            if start_choice == 0:
                lane_from, lane_to, lane_id = 'a', 'b', 0
            elif start_choice == 1:
                lane_from, lane_to, lane_id = 'a', 'b', 1
            elif start_choice == 2:
                lane_from, lane_to, lane_id = 'j', 'k', 0
            else: assert(False)

            vehicle = Vehicle.create_random(
                self,
                lane_from = lane_from,
                lane_to = lane_to,
                lane_id = lane_id,
                speed=self.config["vehicle"]["controlled_speed"],
                spacing=self.config["vehicle"]["controlled_spacing"][v_id],
                color = color[v_id % len(color)]
            )
            self.controlled_vehicle.append(vehicle)

        for v_id in range(self.config["vehicle"]["controlled_vehicle"]-1, -1, -1):
            target_choice = self.config["vehicle"]["controlled_target"][v_id] if self.config["vehicle"]["controlled_target"] else self.np_random.choice(3)

            vehicle = self.controlled_vehicle[v_id]
            if target_choice == 0:
                target, target_id = 'd', 0
            elif target_choice == 1:
                target, target_id = 'd', 1
            elif target_choice == 2:
                target, target_id = 'n', 0
            else: assert(False)
            vehicle.plan_route(target, target_id)


        for v_id in range(0, self.config["vehicle"]["uncontrolled_vehicle"]):
            vehicle = Vehicle.create_random(
                self,
                lane_from = 'a',
                lane_to = 'b',
                speed=5,
                spacing=self.config["vehicle"]["uncontrolled_spacing"],
            )
            self.uncontrolled_vehicle.append(vehicle)

register(
    id='merge-v0',
    entry_point='highway.env:MergeEnv',
)
