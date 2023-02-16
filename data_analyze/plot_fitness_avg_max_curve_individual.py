#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 4 18:47:58 2022

@author: LJ
"""

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import scipy.stats as st

path = "/Users/lj/revolve2-Alife"

# Read files
df_origin = pd.read_csv(path + "/databases_eval1000/data_analysis_1000.csv")
robots = df_origin['robot'].unique()

for i, robot in enumerate(robots):
    df = df_origin
    df = df[df['robot']==robot]

    # Filter values in a column
    # CPG_NES = df[df['controller+learner']=='CPG+NES']
    CPG_RevDE = df[df['controller+learner']=='CPG+RevDE']
    ANN_RevDE = df[df['controller+learner']=='ANN+RevDE']
    DRL_PPO = df[df['controller+learner']=='DRL+PPO']

    # Set parameters
    data = [ANN_RevDE, DRL_PPO, CPG_RevDE] #, CPG_NES
    # color = ['limegreen', 'mediumpurple', 'turquoise', 'deepskyblue']
    # std_color = ['greenyellow', 'mediumpurple', 'aquamarine', 'lightskyblue']

    color = ['mediumpurple', 'deepskyblue', 'darkolivegreen']  # ,'#183125'
    std_color = ['mediumpurple', 'lightskyblue', 'aquamarine']  # ,'aquamarine'

    # Plot and save
    figure, ax = plt.subplots()
    for i, file in enumerate(data):
        describe = data[i].groupby(['nr_evaluations']).describe()['fitness(cm/s)']
        # print(describe)
        mean = describe['mean']
        std = describe['std']
        max = describe['max']
        standard_error = std / np.sqrt(np.size(describe))
        confidence_interval = st.t.interval(confidence=0.95, df=len(describe) - 1, loc=mean, scale=standard_error)

        ax.plot(data[i]['nr_evaluations'].unique(), mean, color=color[i], label=data[i]['controller+learner'].unique()[0], linewidth=2.0)

        # standard deviation
        # plt.fill_between(data[i]['nr_evaluations'].unique(), mean - std, mean + std, color=std_color[i], alpha=0.2)

        # standard error
        # plt.fill_between(data[i]['nr_evaluations'].unique(), mean - standard_error, mean + standard_error, color=std_color[i], alpha=0.2)

        # 95% confidence interval
        plt.fill_between(data[i]['nr_evaluations'].unique(), confidence_interval[0], confidence_interval[1], color=std_color[i], alpha=0.2)

        # max
        ax.scatter(data[i]['nr_evaluations'].unique(), max, s=8, color=color[i])

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.set_xlabel('no. of evaluations')
    ax.set_ylabel('fitness(cm/s)')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0),
              ncol=1, fontsize=9)
    # ax.grid()
    plt.figure(figsize=(3, 100))
    figure.savefig(path+"/databases_eval1000/plot_images/individual/fitness_avg_max_lineplot_"+robot+".png", bbox_inches='tight')

    plt.show()