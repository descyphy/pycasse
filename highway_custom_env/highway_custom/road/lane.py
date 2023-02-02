import numpy as np
from typing import Tuple, Optional
from highway_custom.util import wrap_to_pi


class LineType:
    """A lane side line type."""
    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2

class AbstractLane(object):
    """A lane on the road, described by its central curve."""

    DEFAULT_LANE_WIDTH: float = 6

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        """
        Convert local lane coordinates to a world position.

        :param longitudinal: longitudinal lane coordinate [m]
        :param lateral: lateral lane coordinate [m]
        :return: the corresponding world position [m]
        """
        raise NotImplementedError()

    def coordinate(self, position: np.ndarray) -> Tuple[float, float]:
        """
        Convert a world position to local lane coordinates.

        :param position: a world position [m]
        :return: the (longitudinal, lateral) lane coordinates [m]
        """
        raise NotImplementedError()

    def heading_at(self, longitudinal: float) -> float:
        """
        Get the lane heading at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane heading [rad]
        """
        raise NotImplementedError()

    def width_at(self, longitudinal: float) -> float:
        """
        Get the lane width at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane width [m]
        """
        raise NotImplementedError()

    def after_end_by_position(self, position: np.ndarray) -> bool:
        """
        Whether a given world position is on the lane.

        :param position: a world position [m]
        :param margin: (optional) a supplementary margin around the lane width
        :return: is the position on the lane?
        """
        longitudinal, _ = self.coordinate(position)
        #  print(position)
        #  print(longitudinal)
        #  print(self.length)
        return longitudinal > self.length

    def before_start_by_position(self, position: np.ndarray) -> bool:
        """
        Whether a given world position is on the lane.

        :param position: a world position [m]
        :param margin: (optional) a supplementary margin around the lane width
        :return: is the position on the lane?
        """
        longitudinal, _ = self.coordinate(position)
        return longitudinal < 0

    def on_lane_by_position(self, position: np.ndarray, margin: float = 0) -> bool:
        """
        Whether a given world position is on the lane.

        :param position: a world position [m]
        :param margin: (optional) a supplementary margin around the lane width
        :return: is the position on the lane?
        """
        longitudinal, lateral = self.coordinate(position)
        return self.on_lane_by_coordinate(longitudinal, lateral, margin)

    def on_lane_by_coordinate(self, longitudinal: float, lateral: float, margin: float = 0) -> bool:
        """
        Whether a given coordinate is on the lane.

        :param longitudinal: (optional) the corresponding longitudinal lane coordinate, if known [m]
        :param lateral: (optional) the corresponding lateral lane coordinate, if known [m]
        :param margin: (optional) a supplementary margin around the lane width
        :return: is the position on the lane?
        """
        return np.abs(lateral) <= self.width_at(longitudinal) / 2 + margin and 0 <= longitudinal <= self.length

    def distance_with_heading(self, position: np.ndarray, heading: Optional[float], heading_weight: float = 1.0):
        """Compute a weighted distance in position and heading to the lane."""
        if heading is None:
            return self.distance_by_position(position)
        s, r = self.coordinate(position)
        angle = np.abs(wrap_to_pi(heading - self.heading_at(s)))
        return self.distance_by_coordinate(s,r) + heading_weight*angle

    def distance_by_position(self, position: np.ndarray):
        """Compute the L1 distance [m] from a position to the lane."""
        s, r = self.coordinate(position)
        return self.distance_by_coordinate(s, r)

    def distance_by_coordinate(self, longitudinal: float, lateral: float) -> bool:
        """Compute the L1 distance [m] from a coordinate to the lane."""
        return abs(lateral) + max(longitudinal - self.length, 0) + max(-longitudinal, 0)

    def region(self, left_margin = 0, right_margin = 0) -> np.ndarray:
        """
        Get the region of the lane.

        :return: [[A, B, C, D, E]] with representation of Ax^2 + Bx + Cy^2 + Dy + E <= 0
        """
        raise NotImplementedError()

class StraightLane(AbstractLane):
    """A lane going in straight line."""

    def __init__(self,
                 start: np.ndarray,
                 end: np.ndarray,
                 width: float = AbstractLane.DEFAULT_LANE_WIDTH,
                 line_type: Tuple[LineType, LineType] = (LineType.STRIPED, LineType.STRIPED),
                 speed_limit: float = 20) -> None:
        """
        New straight lane.

        :param start: the lane starting position [m]
        :param end: the lane ending position [m]
        :param width: the lane width [m]
        :param line_type: the type of lines on both sides of the lane
        """
        self.start = np.array(start)
        self.end = np.array(end)
        self.width = width
        self.line_type = line_type

        self.heading = np.arctan2(self.end[1] - self.start[1], self.end[0] - self.start[0])
        self.length = np.linalg.norm(self.end - self.start)

        self.direction_longitudinal = (self.end - self.start) / self.length
        self.direction_lateral = np.array([-self.direction_longitudinal[1], self.direction_longitudinal[0]])

        self.speed_limit = speed_limit

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return self.start + longitudinal * self.direction_longitudinal + lateral * self.direction_lateral

    def coordinate(self, position: np.ndarray) -> Tuple[float, float]:
        delta = position - self.start
        longitudinal = np.dot(delta, self.direction_longitudinal)
        lateral = np.dot(delta, self.direction_lateral)
        return float(longitudinal), float(lateral)

    def heading_at(self, longitudinal: float) -> float:
        return self.heading

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def region(self, left_margin = 0, right_margin = 0) -> np.ndarray:
        equation = []
        equation.append([0, self.direction_lateral[0], 0, self.direction_lateral[1], -self.direction_lateral.dot(self.start) - self.DEFAULT_LANE_WIDTH / 2 + left_margin])
        equation.append([0, -self.direction_lateral[0], 0, -self.direction_lateral[1], self.direction_lateral.dot(self.start) - self.DEFAULT_LANE_WIDTH / 2 + right_margin])
        equation.append([0, -self.direction_longitudinal[0], 0, -self.direction_longitudinal[1], self.direction_longitudinal.dot(self.start)])
        equation.append([0, self.direction_longitudinal[0], 0, self.direction_longitudinal[1], -self.direction_longitudinal.dot(self.end)])
        return equation

    def __repr__(self):
        res = "StraightLane: start: {}, end: {}, width: {}, line_type: {}, speed_limit: {}\n".format(self.start, self.end, self.width, self.line_type, self.speed_limit)
        return res

class CircularLane(AbstractLane):
    """A lane going in circle arc."""

    def __init__(self,
                 center: np.ndarray,
                 radius: float,
                 start_phase: float,
                 end_phase: float,
                 clockwise: bool = False,
                 width: float = AbstractLane.DEFAULT_LANE_WIDTH,
                 line_type: Tuple[LineType, LineType] = (LineType.STRIPED, LineType.STRIPED),
                 speed_limit: float = 20) -> None:
        super().__init__()
        self.center = center
        self.radius = radius
        self.start_phase = wrap_to_pi(start_phase)
        self.end_phase = wrap_to_pi(end_phase)
        self.direction = -1 if clockwise else 1
        self.width = width
        self.line_type = line_type

        self.length = self.longitudinal(self.end_phase)

        self.speed_limit = speed_limit

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        phi = self.phi(longitudinal)
        return self.center + (self.radius - lateral * self.direction) * np.array([np.cos(phi), np.sin(phi)])

    def coordinate(self, position: np.ndarray) -> Tuple[float, float]:
        delta = position - self.center
        phi = np.arctan2(delta[1], delta[0])
        r = np.linalg.norm(delta)
        longitudinal = self.longitudinal(phi)
        lateral = self.direction*(self.radius - r)
        return longitudinal, lateral

    def heading_at(self, longitudinal: float) -> float:
        phi = self.phi(longitudinal)
        psi = wrap_to_pi(phi + np.pi/2 * self.direction)
        return psi

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def longitudinal(self, phi: float) -> float:
        longitudinal = self.radius * wrap_to_pi((phi - self.start_phase) * self.direction, positive = True)
        return longitudinal

    def phi(self, longitudinal: float) -> float:
        phi = wrap_to_pi(self.direction * longitudinal / self.radius + self.start_phase)
        return phi

    def region(self, left_margin = 0, right_margin = 0) -> np.ndarray:
        equation = []
        equation.append([1, -2 * self.center[0], 1, -2 * self.center[1], self.center[0] ** 2 + self.center[1] ** 2 - (self.radius + self.width / 2 - left_margin) ** 2])
        equation.append([-1, 2 * self.center[0], -1, 2 * self.center[1], - self.center[0] ** 2 - self.center[1] ** 2 + (self.radius - self.width / 2 + right_margin) ** 2])

        start_phase_vector = np.array([-np.sin(self.start_phase), np.cos(self.start_phase)])
        start_phase_eq = [0, -start_phase_vector[0], 0, -start_phase_vector[1], start_phase_vector.dot(self.center)]
        equation.append([i * self.direction for i in start_phase_eq])

        end_phase_vector = np.array([np.sin(self.end_phase), -np.cos(self.end_phase)])
        end_phase_eq = [0, -end_phase_vector[0], 0, -end_phase_vector[1], end_phase_vector.dot(self.center)]
        equation.append([i * self.direction for i in end_phase_eq])
        return equation

    def __repr__(self):
        res = "CircularLane: center: {}, radius: {}, start_phase: {}, end_phase: {}, direction: {}, width: {}, line_type: {}, speed_limit: {}\n".format(self.center, self.radius, self.start_phase, self.end_phase, self.direction, self.width, self.line_type, self.speed_limit)
        return res

if __name__ == "__main__":
    l = CircularLane(np.array([0,0]), 10, np.pi / 2, 0)
    #  l = CircularLane(np.array([0,0]), 10, np.pi, 0, clockwise = True)

