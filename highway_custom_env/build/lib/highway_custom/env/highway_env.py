from gym.envs.registration import register

from highway_custom.env.abstract_env import AbstractEnv
from highway_custom.road.road import Road, RoadNetwork
from highway_custom.vehicle.vehicle import Vehicle
from highway_custom.graphic import VehicleGraphics


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the leftmost lanes and avoiding collisions.
    """

    @staticmethod
    def default_config() -> dict:
        config = AbstractEnv.default_config()
        config.update({
            "road": {
                "lanes_count": 4,
                "length": 100,
                "speed_limit": 10,
                "acceleration_limit": 4
                },
            "vehicle": {
                "controlled_vehicle": 1,
                "controlled_start": [0],
                "controlled_target": [0],
                "controlled_speed": 10,
                "controlled_spacing": [2],
                "uncontrolled_vehicle": 0,
                "uncontrolled_spacing": 1,
                },
            "duration": 40,  # [s]
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [20, 30],
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicle()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(self, network=RoadNetwork.straight_road_network(self.config["road"]["lanes_count"], length = self.config["road"]["length"], speed_limit=self.config["road"]["speed_limit"])) 
        #  print(self.road.network)

    def _create_vehicle(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_per_controlled = self.config["vehicle"]["uncontrolled_vehicle"] // self.config["vehicle"]["controlled_vehicle"]

        self.controlled_vehicle = []
        self.uncontrolled_vehicle = []

        color = (VehicleGraphics.RED, VehicleGraphics.GREEN, VehicleGraphics.BLUE, VehicleGraphics.YELLOW, VehicleGraphics.PURPLE)
        for controlled in range(self.config["vehicle"]["controlled_vehicle"]):
            #  print("controlled: {}".format(controlled))
            lane_id = self.config["vehicle"]["controlled_start"][controlled] if self.config["vehicle"]["controlled_start"] else None
            controlled_vehicle = Vehicle.create_random(
                self,
                lane_id = lane_id,
                speed = self.config["vehicle"]["controlled_speed"],
                spacing = self.config["vehicle"]["controlled_spacing"][controlled],
                color = color[controlled % len(color)]
            )

            lane_id = self.config["vehicle"]["controlled_target"][controlled] if self.config["vehicle"]["controlled_target"] else None
            controlled_vehicle.plan_route("1", lane_id)
            self.controlled_vehicle.append(controlled_vehicle)

            if controlled != self.config["vehicle"]["controlled_vehicle"] - 1:
                num_other = other_per_controlled
            else:
                num_other = self.config["vehicle"]["uncontrolled_vehicle"] - controlled * other_per_controlled
            for other in range(num_other):
                #  print("other: {}".format(other))
                self.uncontrolled_vehicle.append(
                    Vehicle.create_random(self, spacing=1 / self.config["vehicle"]["uncontrolled_spacing"], color=VehicleGraphics.YELLOW)
                )

register(
    id='highway-v1',
    entry_point='highway_custom.env:HighwayEnv',
)
