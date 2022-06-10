import argparse
import ast
import contextlib
import os
import random
import termios
import time
import timeit
import tty
import importlib

import gym

import nle  # noqa: F401
from nle import nethack

@contextlib.contextmanager
def dummy_context():
    yield None

def get_action(env, obs):
    return env.action_space.sample()

def play():
    env = gym.make(
        FLAGS.env,
        savedir=FLAGS.savedir,
        max_episode_steps=FLAGS.max_steps,
        allow_all_yn_questions=True,
        allow_all_modes=True,
    )

    module = FLAGS.run + ".agent"
    name = "Agent"
    agent = getattr(importlib.import_module(module), name)(FLAGS)
    if FLAGS.mode == 'train':
        agent.train(env)
    elif FLAGS.mode == 'test':
        agent.run_episodes(env)

def main():
    parser = argparse.ArgumentParser(description="NLE Play tool.")
    parser.add_argument(
        "--run",
        type=str,
        default="agents.example1",
        help="Select what to run. Defaults to example1.",
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        default="NetHackScore-v0",
        help="Gym environment spec. Defaults to 'NetHackScore-v0'.",
    )
    parser.add_argument(
        "-n",
        "--ngames",
        type=int,
        default=1,
        help="Number of games to be played before exiting. "
        "NetHack will auto-restart if > 1.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1_000_000,
        help="Number of maximum steps per episode.",
    )
    parser.add_argument(
        "--savedir",
        default="nle_data/play_data",
        help="Directory path where data will be saved. "
        "Defaults to 'nle_data/play_data'.",
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Disables env.render()."
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        choices=["human", "full", "ansi"],
        help="Render mode. Defaults to 'human'.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "train"],
        help="Test or train. Defaults to 'test'",
    )

    global FLAGS
    FLAGS = parser.parse_args()

    cm = dummy_context

    with cm():
        if FLAGS.savedir == "args":
            FLAGS.savedir = "{}_{}_{}.zip".format(
                time.strftime("%Y%m%d-%H%M%S"), FLAGS.mode, FLAGS.env
            )
        elif FLAGS.savedir == "None":
            FLAGS.savedir = None  # Not saving any ttyrecs.

        play()


if __name__ == "__main__":
    main()