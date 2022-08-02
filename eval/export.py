import csv
import logging
from datetime import datetime

import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.linalg as la
import seaborn as sns

from . import config

logger = logging.getLogger(__name__)


def export_results(config):
    df = pd.read_csv(config.out_file, names=config.csv_columns)

    # 현재 에이전트 목록에 없으면 제외
    xs = list(config.teams.keys())
    df = df[df["agent"].apply(lambda x: x in xs)]

    # 이번주 실시한 #(runs) 최근 게임만 선택
    df = df[df["run"] > df["run"].max() - config.runs]
    run_start = df["run"].max() - config.runs + 1
    run_end = df["run"].max()

    #
    # 평균 score의 평균(over 5 runs)
    #
    df_mean_score = df[["agent", "mean_score"]].groupby("agent")
    mean_score = df_mean_score.mean()

    #
    # median score의 평균(over 5 runs)
    #
    df_med_score = df[["agent", "median_score"]].groupby("agent")
    med_score = df_med_score.mean()

    #
    # 게임 플레이 시간
    #
    df_play_time = df[["agent", "play_time"]].groupby("agent")
    mean_play_time = df_play_time.mean()

    #
    # TABLE
    #
    kv = {
        "mean score": mean_score,
        "median score": med_score,
        "play time": mean_play_time,
    }
    summary = pd.concat(kv.values(), axis=1, sort=True)
    summary.index.name = "agent"
    summary.columns = kv.keys()
    summary = summary.sort_values(by="median score", ascending=False)

    summary.to_csv(config.summary_dir / "summary.csv")

    #
    # README 파일 업데이트
    #
    write_readme(config, run_start, run_end)

def write_readme(config, run_start, run_end):
    def csv_to_table(filename, title):
        buff = f"""
.. list-table:: {title}
   :header-rows: 1
"""
        with filename.open() as f:
            reader = csv.reader(f)
            for row in reader:
                for i, item in enumerate(row):
                    if i == 0:
                        buff += f"   * - {item}\n"
                    else:
                        if item.replace(".", "", 1).isdigit():
                            # float 확인
                            buff += f"     - {float(item):.3f}\n"
                        else:
                            buff += f"     - {item}\n"
        return buff

    t_current = datetime.now()
    summary_table = csv_to_table(config.summary_dir / "summary.csv", "Summary")

    # README 파일 생성
    with (config.out_dir / "README.rst").open("wt") as f:
        content = f"""
NCF2022 결과
===============
.. list-table:: 진행현황
   :header-rows: 1
 
   * - 시작시간
     - 현재시간
     - 경과시간
     - 게임 번호
   * - {config.start_time.isoformat()}
     - {t_current.isoformat()}
     - {t_current - config.start_time}
     - {run_start}부터 {run_end}까지

**결과 요약**

{summary_table}

- 게임번호 {run_start}부터 {run_end}까지 결과(최근 게임 결과)만 사용함
- mean score: 평균 점수
- median score: 점수의 중간값
- player_time: 평균 게임 플레이 시간
"""
        f.write(content)

if __name__ == "__main__":
    export_results(config)