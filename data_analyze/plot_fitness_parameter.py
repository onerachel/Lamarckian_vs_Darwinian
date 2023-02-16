#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 6 10:49:58 2023

@author: LJ
"""

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import scipy.stats as st

path = "/Users/lj/revolve2-Alife"
pd.set_option('display.max_columns', 10)

# Read files
df = pd.read_csv(path + "/databases_eval1000/MBF_parameter.csv")
print(df.head())

# Plot
fig, axs = plt.subplots(3)
axs[0].scatter(df['nr_controller_params_ANN'], df['ANN+RevDE'], s=8, color='mediumpurple')
axs[1].scatter(df['nr_controller_params_DRL'], df['DRL+PPO'], s=8, color='deepskyblue')
axs[2].scatter(df['nr_controller_params_CPG'], df['CPG+RevDE'], s=8, color='darkolivegreen')

# calculate the regression line and plot
coefficients = np.polyfit(df['nr_controller_params_ANN'], df['ANN+RevDE'], 1)
polynomial = np.poly1d(coefficients)
y_regression = polynomial(df['nr_controller_params_ANN'])
axs[0].plot(df['nr_controller_params_ANN'], y_regression, 'r',  linewidth=0.5)

coefficients = np.polyfit(df['nr_controller_params_DRL'], df['DRL+PPO'], 1)
polynomial2 = np.poly1d(coefficients)
y_regression2 = polynomial2(df['nr_controller_params_DRL'])
axs[1].plot(df['nr_controller_params_DRL'], y_regression2, 'r',  linewidth=0.5)

coefficients = np.polyfit(df['nr_controller_params_CPG'], df['CPG+RevDE'], 1)
polynomial3 = np.poly1d(coefficients)
y_regression3 = polynomial3(df['nr_controller_params_CPG'])
axs[2].plot(df['nr_controller_params_CPG'], y_regression3, 'r',  linewidth=0.5)


plt.rcParams['axes.titley'] = 1.0  # y is in axes-relative coordinates.
plt.rcParams['axes.titlepad'] = -14  # pad is in points.
axs[0].set_title('ANN+RevDE')
axs[1].set_title('DRL+PPO')
axs[2].set_title('CPG+RevDE')


for ax in axs.flat:
    ax.set(xlabel='no. of controller parameters', ylabel='fitness (cm/s)')
    # ax.set_ylim(0, 30)
fig.savefig(path + "/databases_eval1000/plot_images/fitness_MBF_params_subplots.png", bbox_inches='tight')

plt.show()
