
from typing import List, Tuple, Optional
import numpy as np

from highway.road.lane import LineType, AbstractLane, StraightLane

LaneIndex = Tuple[str, str, int]


class RoadNetwork(object):

    def __init__(self):
        self.graph = {}

    def add_lane(self, _from: str, _to: str, lane: AbstractLane) -> None:
        """
        A lane is encoded as an edge in the road network.

        :param _from: the node at which the lane starts.
        :param _to: the node at which the lane ends.
        :param AbstractLane lane: the lane geometry.
        """
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)

    def get_lane(self, index: LaneIndex) -> AbstractLane:
        """
        Get the lane geometry corresponding to a given index in the road network.

        :param index: a tuple (origin node, destination node, lane id on the road).
        :return: the corresponding lane geometry.
        """
        _from, _to, _id = index
        return self.graph[_from][_to][_id]

    def get_closest_lane_index(self, position: np.ndarray, heading: Optional[float] = None) -> LaneIndex:
        """
        Get the index of the lane closest to a world position.

        :param position: a world position [m].
        :param heading: a heading angle [rad].
        :return: the index of the closest lane.
        """
        indexes, distances = [], []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance_with_heading(position, heading))
                    indexes.append((_from, _to, _id))
        return indexes[int(np.argmin(distances))]

    def next_lane_given_next_road(self, _from: str, _to: str, _id: int, next_to: str, position: np.ndarray) -> Tuple[int, float]:
        # If next road has same number of lane, stay on the same lane
        if len(self.graph[_from][_to]) == len(self.graph[_to][next_to]):
            next_id = _id
        # Else, pick closest lane
        else:
            lanes = range(len(self.graph[_to][next_to]))
            next_id = min(lanes,
                          key=lambda l: self.get_lane((_to, next_to, l)).distance_by_position(position))
        return next_id

    def bfs_paths(self, start: str, goal: str) -> List[List[str]]:
        """
        Breadth-first search of all routes from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: list of paths from start to goal.
        """
        if start == goal:
            return [start]
        else:
            assert(start in self.graph)
            queue = [(start, [start])]
            while queue:
                (node, path) = queue.pop(0)
                for _next in set(self.graph[node].keys()) - set(path):
                    if _next == goal:
                        return path + [_next]
                    elif _next in self.graph:
                        queue.append((_next, path + [_next]))

    def shortest_path(self, start: str, goal: str) -> List[str]:
        """
        Breadth-first search of shortest path from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: shortest path from start to goal.
        """
        return self.bfs_paths(start, goal)

    @staticmethod
    def straight_road_network(lanes: int = 4,
                              start: float = 0,
                              length: float = 10000,
                              angle: float = 0,
                              speed_limit: float = 30,
                              net: Optional['RoadNetwork'] = None,
                              nodes_str: Tuple[str, str] = ("0", "1")) -> 'RoadNetwork':
        net = net or RoadNetwork()
        nodes_str = nodes_str
        for lane in range(lanes):
            origin = np.array([start, (lanes - 1 - lane) * StraightLane.DEFAULT_LANE_WIDTH])
            end = np.array([start + length, (lanes - 1 - lane) * StraightLane.DEFAULT_LANE_WIDTH])
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            origin = rotation @ origin
            end = rotation @ end
            line_type = [LineType.CONTINUOUS if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS if lane == lanes - 1 else LineType.NONE]
            net.add_lane(*nodes_str, StraightLane(origin, end, line_type=line_type, speed_limit=speed_limit))
        return net

    def __repr__(self):
        res = ""
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    res += "From: {}, To: {}: \n".format(_from, _to)
                    res += "  {}\n".format(l)
        return res


class Road(object):
    """A road is a set of lanes, and a set of vehicles driving on these lanes."""

    def __init__(self,
                 env:'AbstractEnv',
                 network: RoadNetwork = None) -> None:
        """
        New road.

        :param env: the environment
        :param network: the road network describing the lanes
        """
        self.env = env
        self.network = network

    def next_lane(self, current_index: LaneIndex, position: np.ndarray) -> LaneIndex:
        """
        Get the index of the next lane that should be followed after finishing the current lane.

        - If a plan is available and matches with current lane, follow it.
        - Else, pick next road randomly.
        - If it has the same number of lanes as current road, stay in the same lane.
        - Else, pick next road's closest lane.
        :param current_index: the index of the current target lane.
        :param route: the planned route, if any.
        :param position: the vehicle position.
        :param np_random: a source of randomness.
        :return: the index of the next lane to be followed when current lane is finished.
        """
        _from, _to, _id = current_index
        if _to in self.network.graph[_to]:
            # Compute current projected (desired) position
            long, lat = self.network.get_lane(current_index).coordinate(position)
            projected_position = self.network.get_lane(current_index).position(long, lateral=0)
            # If next route is not known
            lane_list = []
            dist_list = []
            for next_to in self.network.graph[_to].keys():
                next_id = self.network.next_lane_given_next_road(_from, _to, _id, next_to, projected_position)
                distance = self.network.get_lane((_to, next_to, next_id)).distance_by_position(projected_position)
                lane_list.append((next_to, next_id))
                dist_list.append(distance)
            prob = 1 / (np.array(dist_list) + 1)
            prob = prob / np.sum(prob)
            #  print(lane_list)
            #  print(prob)
            #  input()
            _id = self.env.np_random.choice(len(lane_list), p = prob)
            #  _id = np.argmin(dist_list)
            next_to, next_id = lane_list[_id]
            #  print(next_to, next_id)
            #  input()
            return _to, next_to, next_id
        else:
            return None

    def close_vehicles_to(self, vehicle: 'Vehicle', distance: float, count: int = None) -> object:
        compared_vehicle = self.env.controlled_vehicle + self.env.uncontrolled_vehicle

        vehicles = [v for v in compared_vehicle
                    if np.linalg.norm(v.position - vehicle.position) <= distance
                    and v is not vehicle ]

        vehicles = sorted(vehicles, key=lambda v: np.linalg.norm(v.position - vehicle.position))
        if count:
            vehicles = vehicles[:count]
        return vehicles

    def step(self, dt: float) -> None:
        """
        Step the dynamics of each entity on the road.

        :param dt: timestep [s]
        """
        vehicle = self.env.controlled_vehicle + self.env.uncontrolled_vehicle
        for v in vehicle:
            v.step(dt)
        for i, v in enumerate(vehicle):
            for other in vehicle[i+1:]:
                v.check_collision(other, dt)

    def __repr__(self):
        res = "{}\n".format(self.network.__repr__())
        return res
