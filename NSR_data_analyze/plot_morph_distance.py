#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 23 15:40:58 2023

@author: LJ
"""
from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats

path = "/Users/lj/revolve2/NSR_data/"
task = "rotation"  #rotation, point_nav
df = pd.read_csv(path + "best_parent_"+f'{task}'+"_distance.csv")
# df = df[df['run'] == 3].reset_index(drop=True)

# #--------------------------------------------plot fitness & dist correlation--------------------------------------------

# Group the data by "run", "generation", and "experiment"
grouped = df.groupby(['generation_x', 'experiment', 'individual_id_x']).agg({'after_x': 'mean', 'dist': 'mean'}).reset_index()

# extract data1 and data2 from the grouped dataframe
lamarckian = grouped.loc[df['experiment'] == 'Lamarckian+Learning']
darwinian = grouped.loc[df['experiment'] == 'Darwinian+Learning']

# extract the numerical data columns from the grouped dataframe
lamarckian_data = lamarckian[['dist']] #'after_x','dist'
darwinian_data = darwinian[['dist']] #'after_x','dist'

# perform a two-sample t-test to calculate the p-value
t_stat, p_val = stats.ttest_ind(lamarckian_data, darwinian_data)
p_value_formatted = "{:.2f}".format(float(p_val))

# Create a scatter plot with different colors for "Lamarckian+Learning" and "Darwinian+Learning"
ax = sns.scatterplot(data=grouped, x='dist', y='after_x', hue='experiment', style='experiment', palette=['blue','orange'], legend=False) #palette='viridis'

# add regression lines to the plot
sns.lmplot(data=grouped, x='dist', y='after_x', hue='experiment', order=1, ci=None, height=8, aspect=1.5, palette=['blue', 'orange'],legend=False)

# Calculate the correlation between "after_c" and "dist" for each experiment
lamarckian_corr = df[df['experiment'] == 'Lamarckian+Learning'][['after_x', 'dist']].corr().iloc[0, 1]
darwinian_corr = df[df['experiment'] == 'Darwinian+Learning'][['after_x', 'dist']].corr().iloc[0, 1]

# format the correlation scores to two decimal places
lamarckian_corr_formatted = "{:.2f}".format(float(lamarckian_corr))
darwinian_corr_formatted = "{:.2f}".format(float(darwinian_corr))

# Set the title and axis labels
title = f"Rotation (Darwinian r={darwinian_corr_formatted}, Lamarckian r={lamarckian_corr_formatted}, p = {p_value_formatted})"
plt.title(title, fontsize=12)
plt.xlabel('distance',fontsize=12)
plt.ylabel('fitness',fontsize=12)

# Rename the legend labels
plt.legend(title="experiment", loc='upper right')

plt.savefig(path+"plot_images/morph_dist_fitness_"+f'{task}'+".png", bbox_inches='tight')
plt.show()
#
# # rotation: fitness/distance corr: (-0.56 for Lamarckian, -0.59 for Darwinian)
# # point_nav: fitness/distance corr: (-0.53 for Lamarckian, -0.55 for Darwinian)


# #--------------------------------------------plot learning delta & distance--------------------------------------------

# Calculate the correlation between "after_c" and "dist" for each experiment
lamarckian_corr = df[df['experiment'] == 'Lamarckian+Learning'][['learning_delta_x', 'dist']].corr().iloc[0, 1]
darwinian_corr = df[df['experiment'] == 'Darwinian+Learning'][['learning_delta_x', 'dist']].corr().iloc[0, 1]

# Group the data by "run", "generation", and "experiment"
grouped = df.groupby(['generation_x', 'experiment', 'individual_id_x']).agg({'learning_delta_x': 'mean', 'dist': 'mean'}).reset_index()

# extract data1 and data2 from the grouped dataframe
lamarckian = grouped.loc[df['experiment'] == 'Lamarckian+Learning']
darwinian = grouped.loc[df['experiment'] == 'Darwinian+Learning']

# extract the numerical data columns from the grouped dataframe
lamarckian_data = lamarckian[['dist']] #'learning_delta_x', 'dist'
darwinian_data = darwinian[['dist']]

# perform a two-sample t-test to calculate the p-value
t_stat, p_val = stats.ttest_ind(lamarckian_data, darwinian_data)
p_value_formatted = "{:.2f}".format(float(p_val))

# Create a scatter plot with different colors for "Lamarckian+Learning" and "Darwinian+Learning"
sns.scatterplot(data=grouped, x='dist', y='learning_delta_x', hue='experiment', style='experiment', palette=['blue','orange']) #palette='viridis'

# add regression lines to the plot
sns.lmplot(data=grouped, x='dist', y='learning_delta_x', hue='experiment', order=1, ci=None, height=8, aspect=1.5, palette=['blue', 'orange'],legend=False)

# Calculate the correlation between "after_x" and "dist" for each experiment
lamarckian_corr = df[df['experiment'] == 'Lamarckian+Learning'][['learning_delta_x', 'dist']].corr().iloc[0, 1]
darwinian_corr = df[df['experiment'] == 'Darwinian+Learning'][['learning_delta_x', 'dist']].corr().iloc[0, 1]

# format the correlation scores to two decimal places
lamarckian_corr_formatted = "{:.2f}".format(float(lamarckian_corr))
darwinian_corr_formatted = "{:.2f}".format(float(darwinian_corr))

# Set the title and axis labels
title = f"Rotation (Darwinian r={darwinian_corr_formatted}, Lamarckian r={lamarckian_corr_formatted})"  # p = {p_value_formatted}
plt.title(title, fontsize=12)
plt.xlabel('distance',fontsize=12)
plt.ylabel('learning delta',fontsize=12)

# Rename the legend labels
plt.legend(title="experiment", loc='upper right')

plt.savefig(path+"plot_images/morph_dist_learning_delta_"+f'{task}'+".png", bbox_inches='tight')
plt.show()
#
# # #rotation:learning delta/distance corr: (-0.49 for Lamarckian, -0.56 for Darwinian)
# # # point_nav: learning delta/distance corr: (-0.48 for Lamarckian, -0.53 for Darwinian)

# # #--------------------------------------------plot generation & distance--------------------------------------------
# Load the data into a pandas DataFrame
df = pd.read_csv(path + "best_parent_"+f'{task}'+"_distance.csv")

# Group the data by "run", "generation", and "experiment"
# grouped = df.groupby(['generation_x', 'experiment', 'individual_id_x']).agg({'dist': 'mean'}).reset_index()

# Create a scatter plot with different colors for "Lamarckian+Learning" and "Darwinian+Learning"
ax = sns.lineplot(data=df, x='generation_x', y='dist', hue='experiment', palette=['deepskyblue', 'mediumpurple']) #palette='viridis'

# Set the title and axis labels
ax.set_title('Rotation', fontsize=11)
ax.set_xlabel('generation')
ax.set_ylabel('distance')

# Rename the legend labels
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['Darwinian+Learning','Lamarckian+Learning'])

plt.savefig(path+"plot_images/morph_dist_gen_"+f'{task}'+".png", bbox_inches='tight')
plt.show()
