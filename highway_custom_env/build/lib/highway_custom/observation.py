import numpy as np

class ObservationType(object):
    def __init__(self, env: 'AbstractEnv', **kwargs) -> None:
        self.env = env

    def observe(self):
        """Get an observation of the environment state."""
        raise NotImplementedError()

class Observer(ObservationType):

    """Observe the kinematics of nearby vehicles."""

    def __init__(self, env: 'AbstractEnv',
                 vehicles_count: int = 5,
                 absolute: bool = True,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        """
        super().__init__(env)
        self.vehicles_count = vehicles_count
        self.absolute = absolute

    def observe(self, vehicle: 'Vehicle') -> np.ndarray:
        origin = vehicle.position if not self.absolute else None
        # Add ego-vehicle
        data = [vehicle.observe(origin)]
        #  print(data)
        
        # Add nearby traffic
        close_vehicle = self.env.road.close_vehicles_to(vehicle, self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - len(self.env.controlled_vehicle))
        if close_vehicle:
            data.extend([v.observe(origin) for v in close_vehicle])

        # Fill missing rows
        if len(data) < self.vehicles_count:
            data.extend([np.zeros((5,))] * (self.vehicles_count - len(data)))
        #  print(len(data))
        #  print(self.vehicles_count)

        data = np.array(data)
        #  print(data)
        #  input()
        return data
