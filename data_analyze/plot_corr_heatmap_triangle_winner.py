#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 11 13:38:50 2022

@author: LJ
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
sns.set(rc={"figure.figsize": (20, 15)})
sns.set_style("whitegrid")

path = "/Users/lj/revolve2"

# Read files
df = pd.read_csv(path + "/databases_eval580/robotzoo_descriptors.csv")
pd.set_option('display.max_columns', None)
print(df.head())

# Filter on winner
DRL_win=df[df['winner_8'] == 'DRL+PPO']
RevDE_win=df[df['winner_8'] == 'CPG+RevDE']


data = [DRL_win, RevDE_win]
for i, file in enumerate(data):
    winner = data[i]['winner_8'].unique()
    data[i] = data[i][['extremities', 'hinge_count', 'active_hinges_count', 'brick_count',
                       'absolute_size', 'symmetry','width', 'height']] # 'coverage','size', 'proportion', 'width', 'height', 'joints','length_of_limbs', 'extensiveness','limbs',

    # Heatmap for all the numerical data including the target 'fitness'
    # Define the heatmap parameters
    # pd.options.display.float_format = "{:,.2f}".format

    # Define correlation matrix
    corr_matrix = data[i].corr()

    # Select 10 most correlated features from corr_matrix
    ## Replace correlation < |0.03| by 0 for a better visibility
    # corr_matrix[(corr_matrix < 0.03) & (corr_matrix > -0.03)] = 0
    # cols = corr_matrix.nlargest(10, 'absolute_size')['absolute_size'].index
    # corr_matrix = np.corrcoef(data[i][cols].values.T)

    # Set mask to hide the upper half triangle
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Plot the heatmap
    sns.heatmap(corr_matrix,
                cbar=True,
                mask=mask,
                fmt='.2f',
                vmax=1.0,
                vmin=-1.0,
                linewidths=0.1,
                annot_kws={"size": 10},
                annot=True,
                # yticklabels=cols.values,
                # xticklabels=cols.values,
                cmap="viridis")
    plt.title(winner[0]+"-Descriptors Correlation")
    plt.savefig(path + "/databases_eval580/plot_images/heat_map_winner_8_"+winner[0]+".png")
    plt.show()