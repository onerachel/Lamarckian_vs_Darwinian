#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 10 5:39:18 2023

@author: LJ
"""
from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats

path = "/Users/lj/revolve2/NSR_data/"
task = "point_nav"  #rotation, point_nav
df = pd.read_csv(path + "best_parent_"+f'{task}'+".csv")

# Group the data by "run", "generation", and "experiment"
# grouped = df.groupby(['generation_x', 'experiment', 'individual_id_x']).agg({'dist': 'mean'}).reset_index()

df['delta'] = df['after_y'] - df['after_x']

plt.figure(figsize=(8, 6))
# Create a scatter plot with different colors for "Lamarckian+Learning" and "Darwinian+Learning"
ax = sns.lineplot(data=df, x='generation_x', y='delta', hue='experiment', palette=['deepskyblue', 'mediumpurple'], dashes=False)

# Perform independent two-sample t-test
darwinian_data = df[df['experiment'] == 'Darwinian+Learning']['delta']
lamarckian_data = df[df['experiment'] == 'Lamarckian+Learning']['delta']
t_stat, p_value = stats.ttest_ind(darwinian_data, lamarckian_data)

# Set the title and axis labels
ax.set_title(f'Point Navigation (p-value: {p_value:.2e})', fontsize=12)
ax.set_xlabel('generation', fontsize=12)
ax.set_ylabel('fitness delta', fontsize=12)

# Rename the legend labels
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['Darwinian+Learning','Lamarckian+Learning'], loc='upper left')

ax.grid(True, alpha=0.2)
plt.savefig(path+"plot_images/genfit_delta_"+f'{task}'+".png", bbox_inches='tight')
plt.show()
