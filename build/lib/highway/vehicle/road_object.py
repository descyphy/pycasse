import numpy as np
from typing import Tuple, Optional

class RoadObject():
    """
    Common interface for objects that appear on the road.

    For now we assume all objects are rectangular.
    """

    DEFAULT_LENGTH: float = 2  # Object length [m]
    DEFAULT_WIDTH: float = 2  # Object width [m]

    def __init__(self, env: 'AbstractEnv', position: np.ndarray, heading: float = 0, speed: float = 0,
                 color: Optional[Tuple[float, float, float]] = None):
        """
        :param env: the environment
        :param position: cartesian position of object in the surface
        :param heading: the angle from positive direction of horizontal axis
        :param speed: cartesian speed of object in the surface
        """
        self.env = env
        self.position = np.array(position, dtype=np.float)
        self.velocity = speed * np.array([np.cos(heading), np.sin(heading)])

        self.color = color

        self.lane_index = self.env.road.network.get_closest_lane_index(self.position, self.heading)
        self.lane = self.env.road.network.get_lane(self.lane_index)

    def observe(self, origin = None):
        d = np.array([1, self.position[0], self.position[1], self.velocity[0], self.velocity[1]])
        if origin is not None:
            d[1:3] = d[1:3] - origin
        return d

    def lane_distance_to(self, other: 'RoadObject', lane: 'AbstractLane') -> float:
        """
        Compute the signed distance to another object along a lane.

        :param other: the other object
        :param lane: a lane
        :return: the distance to the other other [m]
        """
        return lane.coordinate(other.position)[0] - lane.coordinate(self.position)[0]

    def polygon(self) -> np.ndarray:
        points = np.array([
            [-self.DEFAULT_LENGTH / 2, -self.DEFAULT_WIDTH / 2],
            [-self.DEFAULT_LENGTH / 2, +self.DEFAULT_WIDTH / 2],
            [+self.DEFAULT_LENGTH / 2, +self.DEFAULT_WIDTH / 2],
            [+self.DEFAULT_LENGTH / 2, -self.DEFAULT_WIDTH / 2],
        ]).T
        c, s = np.cos(self.heading), np.sin(self.heading)
        rotation = np.array([
            [c, -s],
            [s, c]
        ])
        points = (rotation @ points).T + np.tile(self.position, (4, 1))
        return points

    def check_collision(self, other: 'RoadObject', dt: float = 0) -> None:
        """
        Check for collision with another vehicle.

        :param other: the other vehicle or object
        :param dt: timestep to check for future collisions (at constant velocity)
        """
        if other is self:
            return

        intersecting, transition = self.__is_colliding(other, dt)
        if intersecting:
            self.crashed = other.crashed = True
            self.position += transition
            other.position -= transition

    def __is_colliding(self, other, dt):
        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > self.DEFAULT_LENGTH:
            return False, np.zeros(2,)
        else:
            # Accurate rectangular check
            return self.check_polygon_intersection(self.polygon(), other.polygon(), self.velocity * dt, other.velocity * dt)

    @staticmethod
    def check_polygon_intersection(a, b, displacement_a, displacement_b):
        #  1. calculate norm vector for all sides
        len_a = len(a)
        len_b = len(b)
        norm = np.empty((len_a + len_b, 2))
        norm[:len_a - 1] = a[1:] - a[:-1]
        norm[len_a - 1] = a[0] - a[-1]
        norm[len_a: -1] = b[1:] - b[:-1]
        norm[-1] = b[0] - b[-1]

        norm[:, [0,1]] = norm[:, [1,0]]
        norm[:, 0] = -norm[:,0]
        norm = norm / np.linalg.norm(norm, axis = 1)[:, None]
        #  print(norm)
        #  input()

        #  2. calculate dot matrix
        a_matrix = a @ norm.T
        b_matrix = b @ norm.T
        #  print(a_matrix)
        #  print(b_matrix)
        #  input()

        #  3. calculate max/ min projection
        max_a_proj = np.max(a_matrix, axis = 0)
        min_a_proj = np.min(a_matrix, axis = 0)
        max_b_proj = np.max(b_matrix, axis = 0)
        min_b_proj = np.min(b_matrix, axis = 0)
        #  print(max_a_proj)
        #  print(min_a_proj)
        #  print(max_b_proj)
        #  print(min_b_proj)
        #  input()

        #  4. calculate distance
        distance = np.maximum(min_a_proj - max_b_proj, min_b_proj - max_a_proj)
        argmax_id = np.argmax(distance)
        #  print(distance)
        #  print(argmax_id)
        #  input()

        if (distance[argmax_id] <= 0):
            d = (a - b).mean(axis=0)
            corrected_norm = norm[argmax_id] if d.dot(norm[argmax_id]) > 0 else -norm[argmax_id]
            translation = (-distance[argmax_id] - corrected_norm.dot(displacement_a - displacement_b)) * corrected_norm
            #  print(distance[argmax_id])
            #  print(corrected_norm)
            #  print(corrected_norm.dot(displacement_a - displacement_b))
            return True, translation

        else:
            return False, None

    @property
    def direction(self) -> np.ndarray:
        return self.velocity / self.speed

    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity)

    @property
    def heading(self) -> float:
        return np.arctan2(self.velocity[1], self.velocity[0])

    def __str__(self):
        res = "{}: position: {}, velocity: {}".format(self.__class__.__name__, self.position, self.velocity)
        return res

    def __repr__(self):
        return self.__str__()

