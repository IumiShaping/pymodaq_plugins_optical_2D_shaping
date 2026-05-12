from ..stopping import StoppingFactory, StoppingBase



@StoppingFactory.register_stop()
class NoneStop(StoppingBase):
    """ Never automatically stops the algorithm """
    def tell_stop(self,
                  iterative_index: int,
                  fitness: float,
                  efficiency: float,
                  *args, **kwargs) -> bool:
        return False


@StoppingFactory.register_stop()
class Iter(StoppingBase):
    """ Automatically stops the algorithm after it reached a given number of iteration """

    params = [
        {'title': 'Niter max:', 'name': 'niter_max', 'type': 'int', 'value': 100}
    ]

    def tell_stop(self,
                  iterative_index: int,
                  fitness: float,
                  efficiency: float,
                  *args, **kwargs) -> bool:
        return iterative_index >= self.settings['niter_max']


@StoppingFactory.register_stop()
class FitnessThreshold(StoppingBase):
    """ Automatically stops the algorithm when its fitness is below a given threshold"""

    params = [
        {'title': 'Fitness:', 'name': 'fitness', 'type': 'float', 'value': 0.1}
    ]

    def tell_stop(self,
                  iterative_index: int,
                  fitness: float,
                  efficiency: float,
                  *args, **kwargs) -> bool:
        return fitness <= self.settings['fitness']


@StoppingFactory.register_stop()
class EfficiencyThreshold(StoppingBase):
    """ Automatically stops the algorithm when its efficiency is above a given threshold and after
    a given number of iteration"""

    params = [
        {'title': 'Efficiency:', 'name': 'efficiency', 'type': 'float', 'value': 0.9},
        {'title': 'Niter max:', 'name': 'niter_max', 'type': 'int', 'value': 100},
    ]

    def tell_stop(self,
                  iterative_index: int,
                  fitness: float,
                  efficiency: float,
                  *args, **kwargs) -> bool:
        return fitness <= self.settings['fitness'] and iterative_index >= self.settings['niter_max']

