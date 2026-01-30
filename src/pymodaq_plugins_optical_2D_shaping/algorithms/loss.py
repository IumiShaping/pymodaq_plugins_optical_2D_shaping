from abc import ABC, abstractmethod
from typing import Callable

import torch
from pyqtgraph.parametertree import Parameter
from torch.nn import MSELoss as TorchMSELoss, SmoothL1Loss as TorchSmoothL1Loss


class LossBase(ABC):

    params: list[dict[str, str]] = []  # definition of the specific parameters needed to compute the loss

    def __init__(self, settings: Parameter):
        self.settings = settings  # attribute used to access specific parameters changed by the user

    @abstractmethod
    def compute_loss(self,
                     field_tested: torch.Tensor,
                     field_target: torch.Tensor) -> torch.Tensor:
        """ Compute the loss by returning a 0D Tensor that will be minimized using minimization algorithm

        To be reimplemented
        """
        ...


class LossFactory:
    """The factory class for creating Algorithm"""

    _registry = {}

    @classmethod
    def register_loss(cls) -> Callable:
        """Class decorator method to register Loss class to the internal registry. Must be used as
        decorator above the definition of a LossBase inherited class.

        The Loss class must implement specific class attributes and methods
        """

        def inner_wrapper(wrapped_class: type[LossBase]) -> type[LossBase]:
            name = wrapped_class.__name__

            if name not in cls._registry:
                cls._registry[name] = wrapped_class
            # Return wrapped_class
            return wrapped_class

        # Return decorated function
        return inner_wrapper

    @classmethod
    def get_loss(cls, name: str) -> type[LossBase]:
        """Factory command to get registered loss class
        .
        This method gets the appropriate executor class from the registry

        Parameters
        ----------
        name: str
            The name of the class as specified during registration

        Returns
        -------
        an instance of the executor created
        """

        if name not in cls._registry:
            raise ValueError(f".{name} is not a supported Loss.")

        return cls._registry[name]

    @property
    def losses(self):
        return list(self._registry.keys())


@LossFactory.register_loss()
class LeastSquares(LossBase):
    params = []

    def compute_loss(self,
                     field_tested: torch.Tensor,
                     field_target: torch.Tensor,) -> torch.Tensor:
        """ Compute the loss by returning a 0D Tensor that will be minimized using minimization algorithm
        """
        return ((torch.sum(
                    (torch.abs(field_tested) ** 2 -
                     torch.abs(field_target) ** 2)
                    ** 2)))


@LossFactory.register_loss()
class MSELoss(LossBase):
    params = []

    def __init__(self, settings: Parameter):
        super().__init__(settings)

        self._loss = TorchMSELoss()

    def compute_loss(self,
                     field_tested: torch.Tensor,
                     field_target: torch.Tensor,) -> torch.Tensor:
        """ Compute the loss by returning a 0D Tensor that will be minimized using minimization algorithm
        """
        return self._loss(torch.abs(field_tested), torch.abs(field_target))


@LossFactory.register_loss()
class SmoothL1Loss(LossBase):
    params = [
        {'title': 'Beta', 'name': 'beta', 'type': 'float', 'value': 0.5},
    ]


    def __init__(self, settings: Parameter):
        super().__init__(settings)

        self._loss = TorchSmoothL1Loss(beta=self.settings['beta'])

    def compute_loss(self,
                     field_tested: torch.Tensor,
                     field_target: torch.Tensor,) -> torch.Tensor:
        """ Compute the loss by returning a 0D Tensor that will be minimized using minimization algorithm
        """
        return self._loss(torch.abs(field_tested), torch.abs(field_target))


@LossFactory.register_loss()
class LeastExponent(LossBase):
    params = [
        {'title': 'Loss exponent', 'name': 'exponent', 'type': 'int', 'value': 4, 'min': 2},
    ]

    def compute_loss(self,
                     field_tested: torch.Tensor,
                     field_target: torch.Tensor,) -> torch.Tensor:
        """ Compute the loss by returning a 0D Tensor that will be minimized using minimization algorithm
        """
        return ((torch.sum(
                    (torch.abs(field_tested) ** 2 -
                     torch.abs(field_target) ** 2)
                    ** self.settings['exponent'])))


@LossFactory.register_loss()
class Chicken(LossBase):
    params = [
        {'title': 'Power exponent', 'name': 'power_exponent', 'type': 'int', 'value': 9, 'min': 2},
        {'title': 'Sum exponent', 'name': 'sum_exponent', 'type': 'int', 'value': 4, 'min': 2},

    ]

    def compute_loss(self,
                     field_tested: torch.Tensor,
                     field_target: torch.Tensor, ) -> torch.Tensor:
        """ Compute the loss by returning a 0D Tensor that will be minimized using minimization algorithm
        """

        amplitude_normalized = torch.sum(torch.abs(field_tested) * torch.abs(field_target))

        return 10 ** self.settings['power_exponent'] * (
                1 - torch.sum((torch.abs(field_target) * torch.abs(field_tested) / amplitude_normalized) *
                              torch.cos(torch.angle(field_target) - torch.angle(field_tested))))**self.settings['sum_exponent']


@LossFactory.register_loss()
class ChickenArnaud(LossBase):

    def compute_loss(self,
                     field_tested: torch.Tensor,
                     field_target: torch.Tensor, ) -> torch.Tensor:
        """ Compute the loss by returning a 0D Tensor that will be minimized using minimization algorithm
        """

        return torch.sum((torch.abs(field_target - field_tested) ** 2))


@LossFactory.register_loss()
class Phase(LossBase):

    def compute_loss(self,
                     field_tested: torch.Tensor,
                     field_target: torch.Tensor, ) -> torch.Tensor:
        """ Compute the loss by returning a 0D Tensor that will be minimized using minimization algorithm
        """

        return torch.sum((torch.abs(field_target.angle() -
                                    field_tested.angle()) ** 2))


@LossFactory.register_loss()
class LSQAmplitudePhase(LossBase):
    params = [
        {'title': 'Amplitude exponent', 'name': 'amplitude_exponent', 'type': 'int', 'value': 1, 'min': 1},
        {'title': 'Phase exponent', 'name': 'phase_exponent', 'type': 'int', 'value': 1, 'min': 1},

    ]

    def compute_loss(self,
                     field_tested: torch.Tensor,
                     field_target: torch.Tensor, ) -> torch.Tensor:
        """ Compute the loss by returning a 0D Tensor that will be minimized using minimization algorithm
        """

        amplitude_normalized = torch.sum(torch.abs(field_tested) * torch.abs(field_target))

        return ((torch.sum(
                    torch.abs((torch.abs(field_tested) ** 2 -
                     torch.abs(field_target) ** 2))
                    ** self.settings['amplitude_exponent'])) *
                ((torch.sum(
                    torch.abs(torch.angle(field_tested) ** 2 -
                     torch.angle(field_target) ** 2)
                    ** self.settings['phase_exponent'])))
                )
