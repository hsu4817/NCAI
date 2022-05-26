import random

from ExampleAgent import ExampleAgent

class Agent(ExampleAgent):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

    def get_action(self, env, obs):
        screen = obs['tty_chars']
        if self.is_more(screen):
            action = 0
        elif self.is_yn(screen):
            action = 8
        elif self.is_locked(screen):
            action = 20
        else:
            action = random.randint(1, 16)
        
        return action