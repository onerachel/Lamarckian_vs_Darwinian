#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 4 18:47:58 2023

@author: LJ
"""

import pandas as pd

pd.options.display.max_columns = 100

# Prepare the data data
path = "/Users/lj/revolve2-Alife/databases_eval1000/"

data = ["ANN+RevDE", "CPG+RevDE", "DRL+PPO"]
for i, dat in enumerate(data):
    df = pd.read_csv(path + f"database/{data[i]}/summary.csv")
    # CPG_RevDE = pd.read_csv(path+"CPG_RevDE/summary.csv")
    # DRL_PPO = pd.read_csv(path+"DRL_PPO/summary.csv")

    df['fitness(cm/s)'] = df['fitness'] * 100 / 30
    df['controller+learner'] = data[i]

    df['nr_evaluations'] = df.groupby(['run', 'robot']).cumcount() + 1
    df = df.iloc[:, -5:]
    data[i] = df
df = pd.concat(data[0:3], ignore_index=True)
df = df[df['nr_evaluations'].isin(
    [10, 40, 70, 100, 130, 160, 190, 220, 250, 280, 310, 340, 370, 400, 430, 460, 490, 520, 550, 580, 610, 640, 670,
     700, 730, 760, 790, 820, 850, 880, 910, 940, 970, 1000])]
df.to_csv(path + "/data_analysis_1000_2.csv", index=False)
