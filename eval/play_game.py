import argparse
import datetime
import importlib
import json
import logging
import os
import re
import shlex
import subprocess
import traceback
from types import SimpleNamespace
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_play_game(agent, env, seed, timeout, verbose):

    # agent example -> agents.my_agent
    save_dir = "nle_data/play_data"
    cmd = f"python -m eval.play_game --agent={agent} --env={env} --seed={seed} --savedir={save_dir}"

    result = {}
    log_buff = [f"{cmd}\n\n"]
    ttyrec = None
    try:
        # ttyrec 파일 모두 삭제
        [p.unlink() for p in Path(save_dir).glob("*.bz2")]

        pout = subprocess.run(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )

        ttyrec = [p for p in Path(save_dir).glob("*.bz2")][0]

        stdout_line = pout.stdout.split(b"\n")
        stdout_line = [line.rstrip().decode("utf-8") for line in stdout_line]
        stderr_line = pout.stderr.split(b"\n")
        stderr_line = [line.rstrip().decode("utf-8") for line in stderr_line]
        lines = ["## STDOUT ##\n"] + stdout_line + ["\n## STDERR ##\n"] + stderr_line
        log_buff += lines

        if len(stdout_line) > 1:
            result = json.loads(stdout_line[-2])

    except subprocess.TimeoutExpired:
        # 비정상적으로 게임이 종료된 경우
        # 시간초과
        log_buff += ["\nTimeout Expired\n"]

    except json.decoder.JSONDecodeError:
        breakpoint()

    return result, log_buff, ttyrec


def play_game(args):
    # agent 초기화
    try:
        module = args.agent
        name = "Agent"
        agent = getattr(importlib.import_module(module), name)(args)
        result = agent.evaluate(args.env, args.savedir, args.max_steps, args.seed)

    # except (AttributeError, ImportError):
    except Exception as e:
        import traceback

        logger.error(f"Agent 클래스를 임포트 할 수 없음: {module}, {e}")
        traceback.print_exc()
        result = json.dumps({})
    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser("NC Fellowship 2022-play_game")
    parser.add_argument(
        "--agent",
        type=str,
    )
    parser.add_argument(
        "--env",
        type=str,
        default="NetHackScore-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--savedir",
        default="nle_data/play_data",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "train"],
        help="Test or train. Defaults to 'test'",
    )

    args = parser.parse_args()

    result = play_game(args)
