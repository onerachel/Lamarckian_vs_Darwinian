#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 2 18:47:58 2023
@author: LJ
"""

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.stats import ttest_ind

path = "/Users/lj/revolve2"
task = "point_nav"  # point_nav, rotation

# Read files
df = pd.read_csv(path + "/NSR_data/summary_" + f'{task}' + "_4.csv")
print(df['experiment'].unique())

df['generation'] = df['generation'] + 1

# Filter values in a column
Darwinian_Learning_fixed = df[df['experiment'] == 'Darwinian+Learning (Fixed Body)']
Darwinian_Learning_evolved = df[df['experiment'] == 'Darwinian+Learning (Evolved Body)']
Lamarckian_Learning_fixed = df[df['experiment'] == 'Lamarckian+Learning (Fixed Body)']
Lamarckian_Learning_evolved = df[df['experiment'] == 'Lamarckian+Learning (Evolved Body)']

# Calculate the p-value using Welch's t-test
_, p_value = ttest_ind(Lamarckian_Learning_evolved['learning_delta'], Darwinian_Learning_evolved['learning_delta'], equal_var=False)
print("p-value: ", p_value)

# Set parameters
data = [Darwinian_Learning_fixed, Darwinian_Learning_evolved, Lamarckian_Learning_fixed, Lamarckian_Learning_evolved]

color = ['cyan', 'deepskyblue', 'orchid', 'mediumpurple']
# color = ['deepskyblue', 'mediumpurple', 'darkolivegreen', 'limegreen']

std_color = ['lightskyblue', 'lightskyblue', 'purple', 'mediumpurple']
# std_color = ['lightskyblue', 'mediumpurple', 'aquamarine', 'greenyellow']

# Plot and save
figure, ax = plt.subplots()
for i, file in enumerate(data):
    describe = data[i].groupby(['generation']).describe()['learning_delta']
    mean = describe['mean']
    std = describe['std']
    max = describe['max']

    standard_error = std / np.sqrt(np.size(describe))
    confidence_interval = st.t.interval(confidence=0.95, df=len(describe) - 1, loc=mean, scale=standard_error)

    ax.plot(data[i]['generation'].unique(), mean, color=color[i], label=data[i]['experiment'].unique()[0], linewidth=2.0)

    plt.fill_between(data[i]['generation'].unique(), confidence_interval[0], confidence_interval[1], color=std_color[i], alpha=0.2)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
ax.xaxis.label.set_size(12)
ax.yaxis.label.set_size(12)
ax.set_xlabel('generation')
ax.set_ylabel('learning delta')
ax.legend(loc='upper left', ncol=1, fancybox=True, shadow=True, fontsize=8)
ax.set_title(f'Point Navigation (p-value: {p_value:.1e})')
ax.grid(True, alpha=0.3)
figure.savefig(path + "/NSR_data/plot_images/fitness_learning_delta_" + f'{task}' + "_4.png", bbox_inches='tight')

plt.show()