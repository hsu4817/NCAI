import random

from ExampleAgent import ExampleAgent

class Agent(ExampleAgent):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

    def get_action(self, env, obs):
        action = random.randint(1, 16)
        return action