import gym

from ExampleAgent import ExampleAgent

class Agent(ExampleAgent):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

        raise NotImplementedError('Should implement init function')