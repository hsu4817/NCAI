import itertools
import logging
import os
import shlex
import shutil
import subprocess
import time
import traceback
from datetime import datetime

import pandas as pd
import yaml
from termcolor import colored
from tqdm import tqdm, trange

from eval.export import export_results
from eval.play_game import run_play_game

from . import config
from .config import args

logger = logging.getLogger(__name__)


def update_team_repo(config):
    for name, team in config.teams.items():
        user_id = team.user_id
        repo_name = team.repo_name
        out_dir = team.temp_dir

        logger.info(f"{name}: {user_id}/{repo_name}")
        if not out_dir.exists():
            #Github CLI 설치 필요
            cmd = f"gh repo clone {user_id}/{repo_name} {out_dir}"
            logger.info(f"RUN {cmd}")
            ret = subprocess.call(shlex.split(cmd))
            logger.info(f"{name} 저장소 클론 완료")
        else:
            ret = 0
            logger.warning(f"{name} 저장소 클론 생략")

        if ret == 0:
            agent_path = out_dir / config.agent_path_file
            if agent_path.exists():
                with agent_path.open() as f:
                    content = f.read().replace(":", ": ")
                    name_agent_class = yaml.load(content, yaml.FullLoader)
                if name not in name_agent_class:
                    logger.error(f"'{name}'가 {name_agent_class}에 없음")
                    logger.error(
                        f"config.py의 team_repo_url와 git 저장소의 bot_path.yaml가 일치하는지 확인 필요"
                    )
                    exit(1)
                agent_file = team.temp_dir / name_agent_class[name].strip()
                agent_class = f"{agent_file.parent}/{agent_file.stem}".replace("/", ".")
                team.class_path = agent_class
                logger.info(f"{name} 업데이트 성공")
            else:
                logger.error(f"{name} class_path 없음")
                exit(1)
        else:
            logger.error(f"{name} 저장소 클론 실패")
            exit(1)
        time.sleep(1)
    return config


def play_games(config, run_start, run_end, verbose):
    config.data_dir.mkdir(exist_ok=True, parents=True)

    # 게임 생성
    team_list = config.teams

    # 게임 실행
    for n in trange(run_start, run_end):
        for team in tqdm(team_list, leave=False):
            agent = config.teams[team].class_path

            log_path = config.data_dir / f"{team}-{n}.log"
            log_path.parent.mkdir(exist_ok=True, parents=True)

            map_name = config.args.map_name
            use_lstm = config.args.use_lstm
            timeout = config.args.timeout

            start = time.monotonic()
            try:
                result, log_buff = run_play_game(
                    agent, map_name, use_lstm, timeout, verbose,
                )

                if verbose:
                    color = ("grey", "on_yellow")
                    tqdm.write(
                        colored(
                            f"Run: {n}, {agent}: {result[0]}, {result[1]}",
                            *color,
                        )
                    )

            except Exception as e:
                result = [0.0, 0.0]
                with open(config.system_log_file, "at") as f:
                    f.write(f"{agent}\n")
                    f.write(f"{traceback.format_exc()}\n")

            play_time = int(time.monotonic() - start)

            # 결과 기록
            log_buff += ["\n## RESULT ##\n"]
            log_buff += [
                ",".join(
                    map(
                        str,
                        [
                            agent,
                            n,
                            map_name,
                            result[0],
                            result[1],
                            play_time,
                            datetime.now().isoformat(),
                        ],
                    )
                )
            ]
            log_path.write_text("\n".join(log_buff))

            time.sleep(3)

def write_out_file(config):
    config.out_file.parent.mkdir(parents=True, exist_ok=True)
    data_dir = config.out_dir / config.out_dir.name

    lines = []
    for log in data_dir.glob("**/*.log"):
        last_line = log.read_text().splitlines()[-1]
        _, t = last_line.rsplit(",", 1)
        lines.append((datatime.fromisoformat(t), last_line))
    lines = sorted(lines)
    lines = [line for _, line in lines]
    config.out_file.write_text("\n".join(lines))

def merge_result(config):
    if config.system_log_file.exists():
        dst = config.out_dir / config.system_log_file.name
        dst.write_text(config.system_log_file.read_text())
    
    write_out_file(config)

if __name__ == "__main__":

    """
    # 플레이, 분석, 결과 출력
    python -m eval.run --runs=5
    # 플레이 안하고, 결과 분석만 실행
    python -m eval.run --play_games=false --export_results=true
    """

    logger.info(f"평가 시작: {config.log_id}")

    # 기존 게임결과 로드 또는 초기파일 생성
    write_out_file(config)
    df = pd.read_csv(config.out_file, names=config.csv_columns)

    # 현재 에이전트 목록에 없으면 기존 결과에서 제외
    xs = list(config.teams.keys())
    df = df[df["agent"].apply(lambda x: x in xs)]

    if args.play_games:
        # if args.update_team_repo:
        logger.info(f"에이전트 업데이트")
        config = update_team_repo(config)

        # 게임 시작/종료 번호 식별
        run_start = 0 if len(df) == 0 else df["run"].max() + 1
        run_end = run_start + args.runs
        
        # 평가 시작
        logger.info(f"평가 시작 {run_start} <= run < {run_end}")
        play_games(config, run_start, run_end, verbose=config.verbose)
        merge_result(config)

    if args.export_results:
        logger.info(f"결과 분석 및 문서 생성")
        export_results(config)

    logger.info(f"평가 종료: {config.log_id}")