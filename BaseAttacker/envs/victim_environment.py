from abc import abstractmethod
from envs.environment import Environment

class VictimEnvironment(Environment):
    def __init__(self):
        super(VictimEnvironment, self).__init__()
        self._env_dynamics = None

    @abstractmethod
    def get_dynamics(self):
        """
        Get environment dynamics information, i.e. altitude.
        """
        pass

    @abstractmethod
    def env_system_model(self):
        """
        Get continuous environment dynamics information.
        Could return state transition functions, system parameters,
        or other relevant dynamics information.
        """
        pass
