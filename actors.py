from decision_engine import Brain
from from_lux.game_objects import *
from abc import ABC, abstractmethod
class main_actor():
    def __init__(self,units :dict ,brain:Brain):
        self.units=units
        brain=brain


class Actors(ABC):
    @abstractmethod
    def can_act(self):
        pass

    @abstractmethod
    def action_mask(self):
        pass

    @abstractmethod
    def pick_action(self):
        pass
class units(Actors,Unit):




