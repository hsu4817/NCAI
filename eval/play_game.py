import argparse
import importlib
import json
import logging
import shlex
import subprocess
import timeit
from pathlib import Path

import gym

from ExampleAgent import ExampleAgent

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
        result["etime"] = timeout

    except json.decoder.JSONDecodeError:
        breakpoint()

    return result, log_buff, ttyrec


def evaluate(agent: ExampleAgent, env_name: str, save_dir, max_steps, seed, timeout=60):

    env = gym.make(
        env_name,
        actions=agent.task_actions,
        savedir=save_dir,
        max_episode_steps=max_steps,
        allow_all_yn_questions=True,
        allow_all_modes=True,
    )

    env.seed(seed, seed)
    obs = env.reset()

    steps = 0
    episodes = 0
    reward = 0.0
    action = None

    mean_sps = 0
    mean_reward = 0.0

    get_score = lambda obs_: float(obs_["blstats"][9])

    total_start_time = timeit.default_timer()
    start_time = total_start_time
    check = (start_time, get_score(obs))
    error = False

    while True:
        score = get_score(obs)

        action = agent.get_action(env, obs)

        if action is None:
            error = True
            break

        obs, reward, done, info = env.step(action)
        steps += 1

        if done:
            print("> 환경종료")

        if steps > max_steps:
            print("> 최대 step 초과")
            error = True
            done = True

        if timeit.default_timer() - check[0] > timeout:
            if check[1] >= score:
                # 현재 점수가 timeout안에 갱신되지 않으면 종료
                print(f"> {check[0]-start_time:.2f}초 이후 점수가 증가하지 않아 종료")
                error = True
                done = True

        if done:
            time_delta = timeit.default_timer() - start_time
            sps = steps / time_delta
            break

        if score > check[1]:
            check = (timeit.default_timer(), score)

    env.close()

    etime = timeit.default_timer() - total_start_time
    result = json.dumps(dict(etime=etime, sps=sps, score=score, error=error))
    print(result)
    return result


def play_game(args):
    # agent 초기화
    try:
        module = args.agent
        name = "Agent"
        agent = getattr(importlib.import_module(module), name)(args)
        result = evaluate(agent, args.env, args.savedir, args.max_steps, args.seed)

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
