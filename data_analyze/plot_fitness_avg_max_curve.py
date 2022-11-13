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

path = "/Users/lj/revolve2"

# Read files
df = pd.read_csv(path+"/databases_eval580/data_analysis_580.csv")
# print(df['controller+learner'].unique())

# Filter values in a column
CPG_NES = df[df['controller+learner']=='CPG+NES']
CPG_RevDE = df[df['controller+learner']=='CPG+RevDE']
ANN_RevDE = df[df['controller+learner']=='ANN+RevDE']
DRL_PPO = df[df['controller+learner']=='DRL+PPO']

# Set parameters
data = [ANN_RevDE, CPG_RevDE, CPG_NES, DRL_PPO]
# color = ['limegreen', 'mediumpurple', 'turquoise', 'deepskyblue']
# std_color = ['greenyellow', 'mediumpurple', 'aquamarine', 'lightskyblue']

# dark purple '#450C54'
# dark blue '#2E718E'

color = ['limegreen', '#450C54', 'turquoise', '#2E718E']
std_color = ['limegreen','#450C54', 'aquamarine', '#2E718E']

# color = ['#DBE317', '#450C54', 'turquoise', '#2E718E']
# std_color = ['#DBE317','#450C54', 'aquamarine', '#2E718E']

# color = ['#DBE317', 'mediumpurple', 'turquoise', 'deepskyblue']
# std_color = ['#DBE317', 'mediumpurple', 'aquamarine', 'lightskyblue']

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

    ax.plot(data[i]['nr_evaluations'].unique(), mean, color=color[i], label=data[i]['controller+learner'].unique()[0])

    # standard deviation
    plt.fill_between(data[i]['nr_evaluations'].unique(), mean - std, mean + std, color=std_color[i], alpha=0.2)

    # standard error
    # plt.fill_between(data[i]['nr_evaluations'].unique(), mean - standard_error, mean + standard_error, color=std_color[i], alpha=0.2)

    # 95% confidence interval
    # plt.fill_between(data[i]['nr_evaluations'].unique(), confidence_interval[0], confidence_interval[1], color=std_color[i], alpha=0.2)

    # max
    ax.scatter(data[i]['nr_evaluations'].unique(), max, s=10, color=color[i])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
ax.xaxis.label.set_size(12)
ax.yaxis.label.set_size(12)
ax.set_xlabel('no of evaluations')
ax.set_ylabel('fitness(cm/s)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=2, fancybox=True, shadow=True, fontsize=9)
# ax.grid()
plt.figure(figsize=(3, 100))
figure.savefig(path+"/databases_eval580/plot_images/fitness_avg_max_lineplot.png", bbox_inches='tight')

plt.show()