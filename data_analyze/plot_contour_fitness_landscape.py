#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 6 21:47:58 2021

@author: LJ
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
path = "/Users/lj/revolve2"

# Read files
df1 = pd.read_csv(path + "/databases_eval580/data_analysis_580.csv")
df2 = pd.read_csv(path + "/databases_eval580/robotzoo_descriptors.csv")
pd.set_option('display.max_columns', None)

# Left join two dataframes
df = pd.merge(df1, df2, on='robot', how='left')

# Filter on learners
CPG_NES = df[df['controller+learner'] == 'CPG+NES']
CPG_RevDE = df[df['controller+learner'] == 'CPG+RevDE']
DRL_PPO = df[df['controller+learner'] == 'DRL+PPO']
# ANN_RevDE = df[df['controller+learner']=='ANN+RevDE']


# measures = ['symmetry', 'absolute_size', 'fitness(cm/s)']
# measures = ['brick_count', 'limbs', 'fitness(cm/s)']

# Plot
plt.figure(figsize=(3.31, 3))
# plt.subplot(1, 3, col_ind+1)
fitnesses = DRL_PPO['fitness(cm/s)']
x = DRL_PPO['limbs']
y = DRL_PPO['brick_count']

x_range = np.arange(min(x), max(x), 0.01) #0.02
y_range = np.arange(min(y), max(y), 0.001) #0.01
xx, yy = np.meshgrid(x_range, y_range, sparse=True)
f = np.zeros((xx.size, yy.size))

sigma = 0.1
for ind in range(len(fitnesses)):
    gx = np.exp(-((x_range - x[ind]) / (max(x) - min(x))) ** 2 / (2 * sigma ** 2))
    gy = np.exp(-((y_range - y[ind]) / (max(y) - min(y))) ** 2 / (2 * sigma ** 2))
    g = np.outer(gx, gy)
    f += g * (fitnesses[ind] - np.sum(fitnesses) / len(fitnesses))

f -= np.min(f)
f /= np.max(f)
opt_range = np.max(fitnesses) - np.min(fitnesses)
f = f * opt_range + np.min(fitnesses)

plt.xlim([min(x), max(x)])
plt.ylim([min(y), max(y)])
plt.contourf(x_range, y_range, f.T) #cmap='jet' #levels=np.arange(0.02, 0.08, 0.005)
# plt.tricontour(x, y, z, levels=5, linewidths=0.5, colors='k')
plt.colorbar()
plt.plot(x, y, 'ko', ms=3)

plt.xlabel('absolute size')
plt.ylabel('symmetry')
plt.grid(True)

plt.tight_layout()
# plt.savefig("Contour_evo_symmetry_size.pdf", bbox_inches='tight')
# plt.savefig("Contour_revdeknn_symmetry_size.pdf", bbox_inches='tight')

plt.show()

# plt.figure(figsize=(0.2, 3))
# x_range = np.arange(min(x), max(x), 0.08) #(0.02, 0.08, 0.005)
# y_range = np.arange(min(y), max(y), 0.01)
# xx, yy = np.meshgrid(x_range, y_range, sparse=True)
# np.ones((xx.size, yy.size)) * y_range
# f = np.ones((xx.size, yy.size)) * y_range
# plt.xlim([0.02, 0.08])
# plt.ylim([0.02, 0.075])
# plt.yticks((0.02, 0.075), ('min', 'max'))
# plt.tick_params(axis='y', direction='out', left=False, labelleft=False, right=True, labelright=True)
# plt.xticks([])
# plt.contourf(x_range, y_range, f.T, levels=np.arange(0.02, 0.08, 0.00005))
# plt.tight_layout()
# plt.savefig("Contour_grad.pdf", bbox_inches='tight')
