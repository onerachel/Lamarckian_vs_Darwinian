#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 3 11:47:58 2023

@author: LJ
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
path = "/Users/lj/revolve2"
task = "rotation"  # point_nav, rotation

# Read files
df = pd.read_csv(path + "/NSR_data/summary_" + f'{task}' + ".csv")

# Filter data based on experiment type
Lamarckian_Learning = df[df['experiment'] == 'Lamarckian+Learning']
Darwinian_Learning = df[df['experiment'] == 'Darwinian+Learning']

experiments = [('Lamarckian+Learning', Lamarckian_Learning), ('Darwinian+Learning', Darwinian_Learning)]

# Define function to create contour plot
def create_contour_plot(experiment_name, data):
    plt.figure(figsize=(3.31, 3))

    fitnesses = data['after']
    x = data['absolute_size']
    y = data['symmetry']

    x_range = np.arange(min(x), max(x), 0.01)
    y_range = np.arange(min(y), max(y), 0.001)
    xx, yy = np.meshgrid(x_range, y_range, sparse=True)
    f = np.zeros((xx.size, yy.size))

    sigma = 0.1
    for ind in range(len(fitnesses)):
        gx = np.exp(-((x_range - x.iloc[ind]) / (max(x) - min(x))) ** 2 / (2 * sigma ** 2))
        gy = np.exp(-((y_range - y.iloc[ind]) / (max(y) - min(y))) ** 2 / (2 * sigma ** 2))
        g = np.outer(gx, gy)
        f += g * (fitnesses.iloc[ind] - np.sum(fitnesses) / len(fitnesses))

    f -= np.min(f)
    f /= np.max(f)
    opt_range = np.max(fitnesses) - np.min(fitnesses)
    f = f * opt_range + np.min(fitnesses)

    plt.xlim([min(x), max(x)])
    plt.ylim([min(y), max(y)])
    # Option 1: without the same color scale
    # contour = plt.contourf(x_range, y_range, f.T)

    # Option2: set same color scale to be between -8 and 64
    levels = np.linspace(-8, 64) #(-0.8, 2.8) for rotation
    contour = plt.contourf(x_range, y_range, f.T, levels=levels)
    ticks = np.arange(-8, 65, 8)  # (-0.8, 2.8, 0.4) Set colorbar ticks from -8 to 64 without decimals, in steps of 8
    plt.colorbar(contour, ticks=ticks)
    plt.plot(x, y, 'ko', ms=2)

    plt.xlabel('absolute size')
    plt.ylabel('symmetry')
    plt.title(experiment_name, fontsize=10)
    plt.grid(True)

    plt.tight_layout()

    # Save plot as PDF file
    plt.savefig(path+f"/NSR_data/plot_images/{task}_{experiment_name}_contour_fitness_descriptor5.png", bbox_inches='tight')


# Create contour plots for each experiment
for experiment_name, data in experiments:
    create_contour_plot(experiment_name, data)

plt.show()