from gym.envs.registration import register
import numpy as np

from highway.env.abstract_env import AbstractEnv
from highway.road.lane import LineType, AbstractLane, StraightLane, CircularLane
from highway.road.road import Road, RoadNetwork
from highway.vehicle.vehicle import Vehicle
from highway.graphic import VehicleGraphics


class IntersectionEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "vehicle": {
                "controlled_vehicle": 1,
                "controlled_target": [1],
                "controlled_speed": 3,
                "controlled_spacing": [0.2],
                "uncontrolled_vehicle": 50,
                "uncontrolled_spacing": 0.5,
                },
            "road": {
                "lane_length": 100,
                "extra_width": 5,
                "num_way": 4,
                "speed_limit": 8,
                "acceleration_limit": 30
                },
            "graphic": {
                "screen_width": 900,
                "screen_height": 900,
                "screen_centering_position": [0.5, 0.5],
                "road_centering_position": [0, 0],
                "scaling": 6.5
                },
        })
        return config

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicle()

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:north | 1:west | 2:south | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_LANE_WIDTH
        lane_length = self.config["road"]["lane_length"]  # [m]
        extra_width = self.config["road"]["extra_width"]

        right_turn_radius = lane_width / 2 + extra_width  # [m}
        left_turn_radius = lane_width * 3 / 2  + extra_width# [m}
        midline_radius = lane_width + extra_width

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

        num_way = self.config["road"]["num_way"]
        for corner in range(num_way):
            angle = corner * np.pi / 2
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array([-lane_width / 2, lane_length + midline_radius])
            end = rotation @ np.array([-lane_width / 2, midline_radius])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_type=[s, c], speed_limit=self.config["road"]["speed_limit"]))
            #  # Right turn
            end = (corner + 1) % 4
            if end < num_way:
                r_center = rotation @ (np.array([-midline_radius, midline_radius]))
                net.add_lane("ir" + str(corner), "il" + str(end),
                             CircularLane(r_center, right_turn_radius, angle, angle - np.pi / 2,
                                      clockwise = True, line_type=[n, c], speed_limit=self.config["road"]["speed_limit"]))
            #  # Left turn
            end = (corner - 1) % 4
            if end < num_way:
                l_center = rotation @ (np.array([midline_radius, midline_radius]))
                net.add_lane("ir" + str(corner), "il" + str(end),
                             CircularLane(l_center, left_turn_radius, angle - np.pi, angle - np.pi / 2,
                                      line_type=[n, n], speed_limit=self.config["road"]["speed_limit"]))
            # Straight
            end = (corner + 2) % 4
            if end < num_way:
                l = [s, n] if corner <= 1 else [n, n]
                start = rotation @ np.array([-lane_width / 2, midline_radius])
                end = rotation @ np.array([-lane_width / 2, -midline_radius])
                net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
                         StraightLane(start, end, line_type=l, speed_limit=self.config["road"]["speed_limit"]))
            #  Exit
            start = rotation @ np.array([lane_width / 2, midline_radius])
            end = rotation @ np.array([lane_width / 2, lane_length + midline_radius])
            net.add_lane("il" + str(corner), "o" + str(corner),
                         StraightLane(start, end, line_type=[n, c], speed_limit=self.config["road"]["speed_limit"]))

        self.road = Road(self, network=net)
        #  print(self.road.network)

    def _make_vehicle(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        num_way = self.config["road"]["num_way"]

        self.controlled_vehicle = []
        self.uncontrolled_vehicle = []

        color = (VehicleGraphics.RED, VehicleGraphics.GREEN, VehicleGraphics.BLUE, VehicleGraphics.YELLOW, VehicleGraphics.PURPLE)
        # Controlled vehicles
        for v_id in range(self.config["vehicle"]["controlled_vehicle"]):
            vehicle = Vehicle.create_random(self,
                lane_from = "o{}".format(v_id % num_way),
                lane_to = "ir{}".format(v_id % num_way),
                speed=self.config["vehicle"]["controlled_speed"],
                spacing=self.config["vehicle"]["controlled_spacing"][v_id],
                color = color[v_id % len(color)]
            )
            self.controlled_vehicle.append(vehicle)

        for v_id in range(self.config["vehicle"]["controlled_vehicle"]-1, -1, -1):
            to_id = self.config["vehicle"]["controlled_target"][v_id] if self.config["vehicle"]["controlled_target"] else self.np_random.choice([i for i in range(num_way) if i != v_id])

            vehicle = self.controlled_vehicle[v_id]
            vehicle.plan_route("o{}".format(to_id))
                             

        for v_id in range(0, self.config["vehicle"]["uncontrolled_vehicle"]):
            vehicle = Vehicle.create_random(self,
                speed=3,
                spacing=self.config["vehicle"]["uncontrolled_spacing"],
                follow = True
            )
            self.uncontrolled_vehicle.append(vehicle)
        #  print(self.controlled_vehicle)
        #  print(self.uncontrolled_vehicle)

register(
    id='intersection-v0',
    entry_point='highway.env:IntersectionEnv',
)
