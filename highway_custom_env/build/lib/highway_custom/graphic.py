from copy import copy, deepcopy
import cv2
import pygame
import numpy as np
from typing import Tuple, List

from highway_custom.road import *
from highway_custom.vehicle import *


class Graphic(object):
    """A viewer to render a highway driving environment."""

    def __init__(self, env: 'AbstractEnv') -> None:
        self.env = env

        self.step = 0
        self.vehicle_history = []

        pygame.init()
        pygame.display.set_caption("Highway-env")
        self.screen = None
        self.surface = None
        self.clock = None

    def record(self) -> None:
        self.vehicle_history.append([])
        self.vehicle_history[-1].append([deepcopy(v) for v in self.env.controlled_vehicle])
        self.vehicle_history[-1].append([deepcopy(v) for v in self.env.uncontrolled_vehicle])
        #  print(self.vehicle_history)

    def display(self, file_path = None, from_start = False) -> None:
        """Display the road and vehicles on a pygame window."""
        panel_size = (self.env.config["graphic"]["screen_width"], self.env.config["graphic"]["screen_height"])
        if file_path is not None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            writer = cv2.VideoWriter(file_path, fourcc, self.env.config["simulation_frequency"], panel_size)

        if from_start:
            self.step = 0

        if self.screen is None:
            self.screen = pygame.display.set_mode(panel_size)
            self.surface = WorldSurface(panel_size, pygame.Surface(panel_size), self.env.config["graphic"]["scaling"], self.env.config["graphic"]["screen_centering_position"])
            self.clock = pygame.time.Clock()

        while self.step < len(self.vehicle_history):
            #  self.surface.focus(self.window_position())
            self.surface.focus(np.array(self.env.config["graphic"]["road_centering_position"]))

            RoadGraphics.display_road(self.env, self.surface)

            if self.env.config["graphic"]["show_trajectory"]:
                step_frequency = self.env.config["simulation_frequency"] // self.env.config["simulation_policy_frequency_ratio"]
                assert(step_frequency >= self.env.config["graphic"]["trajectory_frequency"])
                for s in range(0, self.step, int(step_frequency / self.env.config["graphic"]["trajectory_frequency"])):
                    time = s / step_frequency
                    for v in self.vehicle_history[s][0]:
                        VehicleGraphics.display(v, self.surface, label = "t={:.1f}".format(time))
                    for v in self.vehicle_history[s][1]:
                        VehicleGraphics.display(v, self.surface, label = "t={:.1f}".format(time))

            for i, v in enumerate(self.vehicle_history[self.step][0]):
                VehicleGraphics.display(v, self.surface, label = "#{}".format(i))
                TargetGraphics.display(v, self.surface)

            for v in self.vehicle_history[self.step][1]:
                VehicleGraphics.display(v, self.surface)

            self.screen.blit(pygame.transform.flip(self.surface, False, True), (0, 0))
            self.clock.tick(self.env.config["simulation_frequency"])
            pygame.display.flip()

            self.handle_event()

            self.step += 1

            data = pygame.surfarray.array3d(self.screen)
            data = np.flipud(np.rot90(data))
            if file_path is not None:
                writer.write(cv2.cvtColor(data, cv2.COLOR_RGB2BGR))

        if file_path is not None:
            writer.release() 

    def window_position(self) -> np.ndarray:
        """the world position of the center of the displayed window."""
        if self.vehicle_history[self.step][0]:
            pos = np.array([0.0, 0.0])
            for v in self.vehicle_history[self.step][0]:
                pos += v.position
            return pos / len(self.vehicle_history[self.step][0])
        elif self.vehicle_history[self.step][1]:
            pos = np.array([0.0, 0.0])
            for v in self.vehicle_history[self.step][1]:
                pos += v.position
            return pos / len(self.vehicle_history[self.step][1])
        else:
            return np.array([0.0, 0.0])

    def handle_event(self) -> None:
        """Handle pygame events by forwarding them to the display and environment vehicle."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.surface.handle_event(event)

    def close(self) -> None:
        """Close the pygame window."""
        pygame.quit()

class WorldSurface(pygame.Surface):
    """A pygame Surface implementing a local coordinate system so that we can move and zoom in the displayed area."""

    BLACK = (0, 0, 0)
    GREY = (130, 130, 130)
    GREEN = (50, 200, 0)
    YELLOW = (200, 200, 0)
    WHITE = (255, 255, 255)
    BACKGROUND = (153, 255, 153)

    SCALING_FACTOR = 1.3
    MOVING_FACTOR = 0.1

    def __init__(self, size: Tuple[int, int], surf: pygame.SurfaceType, scaling, centering_position) -> None:
        super().__init__(size, 0, surf)
        self.origin = np.array([0, 0])
        self.scaling = scaling
        self.centering_position = centering_position

    def pix(self, length: float) -> int:
        """
        Convert a distance [m] to pixels [px].

        :param length: the input distance [m]
        :return: the corresponding size [px]
        """
        return int(length * self.scaling)

    def pos2pix(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert two world coordinates [m] into a position in the surface [px]

        :param x: x world coordinate [m]
        :param y: y world coordinate [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pix(x - self.origin[0]), self.pix(y - self.origin[1])

    def vec2pix(self, pos: np.ndarray) -> Tuple[int, int]:
        """
        Convert a world position [m] into a position in the surface [px].

        :param pos: a world position [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pos2pix(pos[0], pos[1])

    def is_visible(self, vec: np.ndarray, margin: int = 50) -> bool:
        """
        Is a position visible in the surface?
        :param vec: a position
        :param margin: margins around the frame to test for visibility
        :return: whether the position is visible
        """
        x, y = self.vec2pix(vec)
        return -margin < x < self.get_width() + margin and -margin < y < self.get_height() + margin

    def focus(self, position: np.ndarray) -> None:
        """
        Set the origin of the displayed area to center on a given world position.

        :param position: a world position [m]
        """
        self.origin = position - np.array(
            [self.centering_position[0] * self.get_width() / self.scaling,
             self.centering_position[1] * self.get_height() / self.scaling])

    def handle_event(self, event: pygame.event.EventType) -> None:
        """
        Handle pygame events for moving and zooming in the displayed area.

        :param event: a pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.scaling *= 1 / self.SCALING_FACTOR
            elif event.key == pygame.K_DOWN:
                self.scaling *= self.SCALING_FACTOR
            if event.key == pygame.K_h:
                self.centering_position[0] -= self.MOVING_FACTOR
            elif event.key == pygame.K_j:
                self.centering_position[1] -= self.MOVING_FACTOR
            elif event.key == pygame.K_k:
                self.centering_position[1] += self.MOVING_FACTOR
            elif event.key == pygame.K_l:
                self.centering_position[0] += self.MOVING_FACTOR
            #  print(self.centering_position)

    @property
    def center(self):
        res = self.origin + np.array([self.get_width() / (2 * self.scaling), self.get_height() / (2 * self.scaling)])
        return res

    @classmethod
    def transparentize(cls, color, transparency = 30):
        assert(len(color) == 3)
        res = (color[0], color[1], color[2], transparency)
        return res

    @classmethod
    def darken(cls, color, ratio=0.83):
        return (
            int(color[0] * ratio),
            int(color[1] * ratio),
            int(color[2] * ratio),
        ) + color[3:]

    @classmethod
    def lighten(cls, color, ratio=0.68):
        return (
            min(int(color[0] / ratio), 255),
            min(int(color[1] / ratio), 255),
            min(int(color[2] / ratio), 255),
        ) + color[3:]


class RoadGraphics(object):
    """A visualization of a road lanes and vehicles."""

    @staticmethod
    def display_road(env: 'AbstractEnv', surface: WorldSurface) -> None:
        """
        Display the road lanes on a surface.

        :param env: the environment
        :param surface: the pygame surface
        """
        surface.fill(surface.BACKGROUND)
        for _from in env.road.network.graph.keys():
            for _to in env.road.network.graph[_from].keys():
                for l in env.road.network.graph[_from][_to]:
                    LaneGraphics.display_road(l, surface)

        for _from in env.road.network.graph.keys():
            for _to in env.road.network.graph[_from].keys():
                for l in env.road.network.graph[_from][_to]:
                    LaneGraphics.display_line(l, surface)

class LaneGraphics(object):
    """A visualization of a lane."""

    STRIPE_SPACING: float = 5
    """ Offset between stripes [m]"""

    STRIPE_LENGTH: float = 3
    """ Length of a stripe [m]"""

    STRIPE_WIDTH: float = 0.5
    """ Width of a stripe [m]"""

    @classmethod
    def display_road(cls, lane: AbstractLane, surface: WorldSurface) -> None:
        #  print(lane)
        stripes_count = int(np.ceil((surface.get_height() + surface.get_width()) / (cls.STRIPE_SPACING * surface.scaling)))
        s_center, _ = lane.coordinate(surface.center)
        s0 = (int(s_center) // cls.STRIPE_SPACING - stripes_count // 2) * cls.STRIPE_SPACING

        if type(lane) is StraightLane:
            cls.continuous_line(lane, surface, stripes_count, s0, 0.5, surface.GREY, lane.width * 1.5)
        elif type(lane) is CircularLane:
            cls.continuous_curve(lane, surface, stripes_count, s0, 0.5, surface.GREY, lane.width * 1.5)

    @classmethod
    def display_line(cls, lane: AbstractLane, surface: WorldSurface) -> None:
        """
        Display a lane on a surface.

        :param lane: the lane to be displayed
        :param surface: the pygame surface
        """
        #  print(lane)
        stripes_count = int(np.ceil((surface.get_height() + surface.get_width()) / (cls.STRIPE_SPACING * surface.scaling)))
        s_center, _ = lane.coordinate(surface.center)
        s0 = (int(s_center) // cls.STRIPE_SPACING - stripes_count // 2) * cls.STRIPE_SPACING

        for side in range(2):
            if lane.line_type[side] == LineType.STRIPED:
                cls.striped_line(lane, surface, stripes_count, s0, side, surface.WHITE, cls.STRIPE_WIDTH)
            elif lane.line_type[side] == LineType.CONTINUOUS and type(lane) is StraightLane:
                cls.continuous_line(lane, surface, stripes_count, s0, side, surface.WHITE, cls.STRIPE_WIDTH)
            elif lane.line_type[side] == LineType.CONTINUOUS and type(lane) is CircularLane:
                cls.continuous_curve(lane, surface, stripes_count, s0, side, surface.WHITE, cls.STRIPE_WIDTH)
            else: assert(lane.line_type[side] == LineType.NONE)

    @classmethod
    def striped_line(cls, lane: AbstractLane, surface: WorldSurface, stripes_count: int, longitudinal: float,
                side: float, color: Tuple, width: float) -> None:
        """
        Draw a striped line on one side of a lane, on a surface.

        :param lane: the lane
        :param surface: the pygame surface
        :param stripes_count: the number of stripes to draw
        :param longitudinal: the longitudinal position of the first stripe [m]
        :param side: which side of the road to draw [0:left, 1:right]
        """
        for s in range(stripes_count):
            start = longitudinal + s * cls.STRIPE_SPACING
            end = longitudinal + s * cls.STRIPE_SPACING + cls.STRIPE_LENGTH
            lat = (0.5 - side) * lane.width_at(start)
            cls.draw_line(lane, surface, start, end, lat, color, width)

    @classmethod
    def continuous_curve(cls, lane: AbstractLane, surface: WorldSurface, stripes_count: int, longitudinal: float,
                side: float, color: Tuple, width: float) -> None:
        """
        Draw a striped line on one side of a lane, on a surface.

        :param lane: the lane
        :param surface: the pygame surface
        :param stripes_count: the number of stripes to draw
        :param longitudinal: the longitudinal position of the first stripe [m]
        :param side: which side of the road to draw [0:left, 1:right]
        """
        for s in range(stripes_count):
            start = longitudinal + s * cls.STRIPE_SPACING
            end = longitudinal + s * cls.STRIPE_SPACING + cls.STRIPE_SPACING
            lat = (0.5 - side) * lane.width_at(start)
            cls.draw_line(lane, surface, start, end, lat, color, width)
        #  print(starts)
        #  print(ends)
        #  print(lats)

    @classmethod
    def continuous_line(cls, lane: AbstractLane, surface: WorldSurface, stripes_count: int, longitudinal: float,
                side: float, color: Tuple, width: float) -> None:
        """
        Draw a continuous line on one side of a lane, on a surface.

        :param lane: the lane
        :param surface: the pygame surface
        :param stripes_count: the number of stripes that would be drawn if the line was striped
        :param longitudinal: the longitudinal position of the start of the line [m]
        :param side: which side of the road to draw [0:left, 1:right]
        """
        start = longitudinal
        end = longitudinal + stripes_count * cls.STRIPE_SPACING + cls.STRIPE_LENGTH
        lat = (0.5 - side) * lane.width_at(start)
        cls.draw_line(lane, surface, start, end, lat, color, width)

    @classmethod
    def draw_line(cls, lane: AbstractLane, surface: WorldSurface,
            start: float, end: float, lat: float, color: Tuple, width: float) -> None:
        """
        Draw a set of stripes along a lane.

        :param lane: the lane
        :param surface: the surface to draw on
        :param start: a starting longitudinal positions for each stripe [m]
        :param end: a ending longitudinal positions for each stripe [m]
        :param lat: a lateral positions for each stripe [m]
        """
        start = np.clip(start, 0, lane.length)
        end = np.clip(end, 0, lane.length)
        pygame.draw.line(surface, color,
                         (surface.vec2pix(lane.position(start, lat))),
                         (surface.vec2pix(lane.position(end, lat))),
                         max(surface.pix(width), 1))

class VehicleGraphics(object):
    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (200, 200, 0)
    BLACK = (60, 60, 60)
    PURPLE = (200, 0, 150)
    DEFAULT_COLOR = YELLOW
    EGO_COLOR = GREEN

    @classmethod
    def display(cls, vehicle: Vehicle, surface: "WorldSurface", transparent: bool = False, label: Optional[str] = None) -> None:
        """
        Display a vehicle on a pygame surface.

        The vehicle is represented as a colored rotated rectangle.

        :param vehicle: the vehicle to be drawn
        :param surface: the surface to draw the vehicle on
        :param transparent: whether the vehicle should be drawn slightly transparent
        :param label: whether a text label should be rendered
        """
        if not surface.is_visible(vehicle.position, vehicle.DEFAULT_LENGTH):
            return

        v = vehicle
        tire_length, tire_width = 1, 0.3
        headlight_length, headlight_width = 0.72, 0.6
        roof_length, roof_width = 2.0, 1.5

        # Vehicle rectangle
        length = v.DEFAULT_LENGTH + 2 * tire_length
        vehicle_surface = pygame.Surface((surface.pix(length), surface.pix(length)),
                                         flags=pygame.SRCALPHA)  # per-pixel alpha
        rect = (surface.pix(tire_length),
                surface.pix(length / 2 - v.DEFAULT_WIDTH / 2),
                surface.pix(v.DEFAULT_LENGTH),
                surface.pix(v.DEFAULT_WIDTH))
        rect_headlight_left = (surface.pix(tire_length+v.DEFAULT_LENGTH-headlight_length),
                               surface.pix(length / 2 - (1.4*v.DEFAULT_WIDTH) / 3),
                               surface.pix(headlight_length),
                               surface.pix(headlight_width))
        rect_headlight_right = (surface.pix(tire_length+v.DEFAULT_LENGTH-headlight_length),
                                surface.pix(length / 2 + (0.6*v.DEFAULT_WIDTH) / 5),
                                surface.pix(headlight_length),
                                surface.pix(headlight_width))
        rect_roof = (surface.pix(v.DEFAULT_LENGTH/2 - tire_length/2),
                     surface.pix(0.999*length/ 2 - 0.38625*v.DEFAULT_WIDTH),
                     surface.pix(roof_length),
                     surface.pix(roof_width))
        color = cls.get_color(v, transparent)
        pygame.draw.rect(vehicle_surface, color, rect, 0)
        pygame.draw.rect(vehicle_surface, WorldSurface.lighten(color), rect_headlight_left, 0)
        pygame.draw.rect(vehicle_surface, WorldSurface.lighten(color), rect_headlight_right, 0)
        pygame.draw.rect(vehicle_surface, WorldSurface.darken(color), rect_roof, 0)
        pygame.draw.rect(vehicle_surface, cls.BLACK, rect, 1)

        # Tires
        tire_positions = [[surface.pix(tire_length), surface.pix(length / 2 - v.DEFAULT_WIDTH / 2)],
                          [surface.pix(tire_length), surface.pix(length / 2 + v.DEFAULT_WIDTH / 2)],
                          [surface.pix(length - tire_length), surface.pix(length / 2 - v.DEFAULT_WIDTH / 2)],
                          [surface.pix(length - tire_length), surface.pix(length / 2 + v.DEFAULT_WIDTH / 2)]]
        angle = np.arctan2(v.action[1], v.action[0])
        tire_angles = [0, 0, angle, angle]
        for tire_position, tire_angle in zip(tire_positions, tire_angles):
            tire_surface = pygame.Surface((surface.pix(tire_length), surface.pix(tire_length)), pygame.SRCALPHA)
            rect = (0, surface.pix(tire_length/2-tire_width/2), surface.pix(tire_length), surface.pix(tire_width))
            pygame.draw.rect(tire_surface, cls.BLACK, rect, 0)
            cls.blit_rotate(vehicle_surface, tire_surface, tire_position, np.rad2deg(-tire_angle))

        # Centered rotation
        h = v.heading if abs(v.heading) > 2 * np.pi / 180 else 0
        position = [*surface.pos2pix(v.position[0], v.position[1])]
        vehicle_surface = pygame.Surface.convert_alpha(vehicle_surface)
        cls.blit_rotate(surface, vehicle_surface, position, np.rad2deg(-h))

        # Label
        if label:
            font = pygame.font.Font(None, 12)
            text = font.render(label, 1, (10, 10, 10), (255, 255, 255))
            position = [*surface.pos2pix(v.position[0], v.position[1] + v.DEFAULT_WIDTH / 2)]
            surface.blit(pygame.transform.flip(text, False, True), position)

    @staticmethod
    def blit_rotate(surf: pygame.SurfaceType, image: pygame.SurfaceType, pos: np.ndarray, angle: float,
                    origin_pos: np.ndarray = None, show_rect: bool = False) -> None:
        """Many thanks to https://stackoverflow.com/a/54714144."""
        # calculate the axis aligned bounding box of the rotated image
        w, h = image.get_size()
        box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]
        min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
        max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

        # calculate the translation of the pivot
        if origin_pos is None:
            origin_pos = w / 2, h / 2
        pivot = pygame.math.Vector2(origin_pos[0], -origin_pos[1])
        pivot_rotate = pivot.rotate(angle)
        pivot_move = pivot_rotate - pivot

        # calculate the upper left origin of the rotated image
        origin = (pos[0] - origin_pos[0] + min_box[0] - pivot_move[0], pos[1] - origin_pos[1] - max_box[1] + pivot_move[1])
        # get a rotated image
        rotated_image = pygame.transform.rotate(image, angle)
        # rotate and blit the image
        surf.blit(rotated_image, origin)
        # draw rectangle around the image
        if show_rect:
            pygame.draw.rect(surf, (255, 0, 0), (*origin, *rotated_image.get_size()), 2)

    @classmethod
    def get_color(cls, vehicle: Vehicle, transparent: bool = False) -> Tuple[int]:
        color = cls.DEFAULT_COLOR
        if vehicle.crashed:
            color = cls.RED
        elif getattr(vehicle, "color", None):
            color = vehicle.color
        if transparent:
            color = WorldSurface.transparentize(color)
        return color
        return (
            min(int(color[0] / ratio), 255),
            min(int(color[1] / ratio), 255),
            min(int(color[2] / ratio), 255),
        ) + color[3:]

class TargetGraphics(object):
    @classmethod
    def display(cls, vehicle: Vehicle, surface: "WorldSurface") -> None:
        """
        Display a target of a vehicle on a pygame surface.

        :param vehicle: the vehicle to be drawn
        :param surface: the surface to draw the vehicle on
        """
        if not surface.is_visible(vehicle.target, vehicle.DEFAULT_LENGTH):
            return

        v = vehicle
        width = 2
        pos = surface.vec2pix(v.target)
        rect = (pos[0] - surface.pix(width / 2),
                pos[1] - surface.pix(width / 2),
                surface.pix(width),
                surface.pix(width))
        color = VehicleGraphics.get_color(v)
        pygame.draw.rect(surface, color, rect, 0)

