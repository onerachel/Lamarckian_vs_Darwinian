#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 6 13:38:50 2022

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
df1 = pd.read_csv(path + "/databases_eval580/data_analysis_580.csv")
df2 = pd.read_csv(path + "/databases_eval580/robotzoo_descriptors.csv")
pd.set_option('display.max_columns', None)
# print(df2.head())
# print(df1.shape)

# Left join two dataframes
df = pd.merge(df1, df2, on='robot', how='left')
# print(df.head())
# print(df.shape)
# print(df_all.nlargest(10, 'fitness(cm/s)'))

# Filter on learners and select top 100 robots based on fitness
CPG_NES = df[df['controller+learner'] == 'CPG+NES'].nlargest(100, 'fitness(cm/s)')
CPG_RevDE = df[df['controller+learner'] == 'CPG+RevDE'].nlargest(100, 'fitness(cm/s)')
DRL_PPO = df[df['controller+learner'] == 'DRL+PPO'].nlargest(100, 'fitness(cm/s)')
ANN_RevDE = df[df['controller+learner'] == 'ANN+RevDE'].nlargest(100, 'fitness(cm/s)')

data = [CPG_NES, CPG_RevDE, ANN_RevDE, DRL_PPO]
for i, file in enumerate(data):
    learner = data[i]['controller+learner'].unique()
    data[i] = data[i][['fitness(cm/s)', 'limbs', 'extremities', 'length_of_limbs', 'extensiveness',
                       'coverage', 'joints', 'hinge_count', 'active_hinges_count', 'brick_count',
                       'proportion', 'width', 'height', 'size', 'absolute_size', 'symmetry']]  ##print(df2.columns)

    # Heatmap for all the numerical data including the target 'fitness'
    # Define the heatmap parameters
    # pd.options.display.float_format = "{:,.2f}".format

    # Define correlation matrix
    corr_matrix = data[i].corr()

    # Select 10 most correlated features from corr_matrix
    ## Replace correlation < |0.03| by 0 for a better visibility
    # corr_matrix[(corr_matrix < 0.03) & (corr_matrix > -0.03)] = 0
    cols = corr_matrix.nlargest(10, 'fitness(cm/s)')['fitness(cm/s)'].index
    corr_matrix = np.corrcoef(data[i][cols].values.T)

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
                yticklabels=cols.values,
                xticklabels=cols.values,
                cmap="viridis")
    plt.title(learner[0]+"-Fitness Correlation")
    plt.savefig(path + "/databases_eval580/plot_images/heat_map_"+learner[0]+".png")
    plt.show()

    # Visualize individually
    corr = data[i].corr()["fitness(cm/s)"].sort_values(ascending=False)[1:7] ## selecting 6 cols other than fitness(cm/s)
    print(learner[0])
    print(corr)
#    f, ax = plt.subplots(nrows=6, ncols=1, figsize=(20, 40))
#    for i, col in enumerate(corr.index):
#        sns.scatterplot(x=col, y="fitness(cm/s)", data=df, ax=ax[i], color='darkorange')
#        ax[i].set_title(f'{col} vs fitness')
#    plt.savefig(path + "/databases_eval580/plot_images/correlation_scatter_plot_"+learner[0]+".png")
#    plt.show()
