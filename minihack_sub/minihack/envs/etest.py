from minihack import MiniHackNavigation
from minihack.envs import register


class MiniHackETest(MiniHackNavigation):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 200)
        super().__init__(*args, des_file="etest.des", **kwargs)



register(
    id="MiniHack-ETest-v0",
    entry_point="minihack.envs.etest:MiniHackETest",
)