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
from scipy.stats import ttest_ind

path = "/Users/lj/revolve2/NSR_data/"
task = "point_nav"  #rotation, point_nav
df = pd.read_csv(path + "summary_"+f'{task}'+".csv")
# df = df[df['run'] == 3].reset_index(drop=True)

# # #--------------------------------------------plot fitness & dist correlation--------------------------------------------
#
# # Group the data by "run", "generation", and "experiment"
# grouped = df.groupby(['generation', 'experiment', 'individual_id']).agg({'after': 'mean', 'distance': 'mean'}).reset_index()
#
# # extract data1 and data2 from the grouped dataframe
# lamarckian = grouped.loc[df['experiment'] == 'Lamarckian+Learning']
# darwinian = grouped.loc[df['experiment'] == 'Darwinian+Learning']
#
# # Filter out rows where "generation" equals 0
# filtered_grouped = grouped[grouped['generation'] != 0]
#
# # Create a scatter plot with different colors for "Lamarckian+Learning" and "Darwinian+Learning"
# ax = sns.scatterplot(data=filtered_grouped, x='distance', y='after', hue='experiment', style='experiment', palette=['blue','orange'], legend=False) #palette='viridis'
#
# # add regression lines to the plot
# sns.lmplot(data=filtered_grouped, x='distance', y='after', hue='experiment', order=1, ci=None, height=6, aspect=1.2, scatter_kws={'s': 15}, palette=['blue', 'orange'],legend=False)
#
# plt.xlim(0.1, 14.5)
# plt.ylim(0, 35)
#
# # Calculate the correlation between "after" and "dist" for each experiment
# filtered_lamarckian_corr = df[(df['experiment'] == 'Lamarckian+Learning') & (df['generation'] != 0)][['after', 'distance']].corr().iloc[0, 1]
# filtered_darwinian_corr = df[(df['experiment'] == 'Darwinian+Learning') & (df['generation'] != 0)][['after', 'distance']].corr().iloc[0, 1]
#
# # Format the correlation scores to two decimal places
# filtered_lamarckian_corr_formatted = "{:.2f}".format(float(filtered_lamarckian_corr))
# filtered_darwinian_corr_formatted = "{:.2f}".format(float(filtered_darwinian_corr))
#
# # Set the title and axis labels
# title = f"Rotation (Darwinian r={filtered_darwinian_corr_formatted}, Lamarckian r={filtered_lamarckian_corr_formatted})" #p = {p_value_formatted}
# plt.title(title, fontsize=12)
# plt.xlabel('distance',fontsize=12)
# plt.ylabel('fitness',fontsize=12)
#
# # Rename the legend labels
# plt.legend(title="", loc='upper right',fontsize=10)
#
# plt.savefig(path+"plot_images/tree_dist_fitness_"+f'{task}'+".png", bbox_inches='tight')
# plt.show()

# # # #--------------------------------------------plot learning delta & distance--------------------------------------------
# #
# # Group the data by "run", "generation", and "experiment"
# grouped = df.groupby(['generation', 'experiment', 'individual_id']).agg({'learning_delta': 'mean', 'distance': 'mean'}).reset_index()
#
# # extract data1 and data2 from the grouped dataframe
# lamarckian = grouped.loc[df['experiment'] == 'Lamarckian+Learning']
# darwinian = grouped.loc[df['experiment'] == 'Darwinian+Learning']
#
# # Filter out rows where "generation" equals 0
# filtered_grouped = grouped[grouped['generation'] != 0]
#
# # Create a scatter plot with different colors for "Lamarckian+Learning" and "Darwinian+Learning"
# ax = sns.scatterplot(data=filtered_grouped, x='distance', y='learning_delta', hue='experiment', style='experiment', palette=['blue','orange'], legend=False) #palette='viridis'
#
# # add regression lines to the plot
# sns.lmplot(data=filtered_grouped, x='distance', y='learning_delta', hue='experiment', order=1, ci=None, height=6, aspect=1.2, scatter_kws={'s': 15}, palette=['blue', 'orange'],legend=False)
#
# plt.xlim(0.1, 14.5)
#
# # Calculate the correlation between "after_c" and "dist" for each experiment
# filtered_lamarckian_corr = df[(df['experiment'] == 'Lamarckian+Learning') & (df['generation'] != 0)][['learning_delta', 'distance']].corr().iloc[0, 1]
# filtered_darwinian_corr = df[(df['experiment'] == 'Darwinian+Learning') & (df['generation'] != 0)][['learning_delta', 'distance']].corr().iloc[0, 1]
#
# # Format the correlation scores to two decimal places
# filtered_lamarckian_corr_formatted = "{:.2f}".format(float(filtered_lamarckian_corr))
# filtered_darwinian_corr_formatted = "{:.2f}".format(float(filtered_darwinian_corr))
#
# # Set the title and axis labels
# title = f"Rotation (Darwinian r={filtered_darwinian_corr_formatted}, Lamarckian r={filtered_lamarckian_corr_formatted})" #p = {p_value_formatted}
# plt.title(title, fontsize=12)
# plt.xlabel('distance',fontsize=12)
# plt.ylabel('learning delta',fontsize=12)
#
# # Rename the legend labels
# plt.legend(title="experiment", loc='upper right')
#
# plt.savefig(path+"plot_images/tree_dist_learning_delta_"+f'{task}'+".png", bbox_inches='tight')
# plt.show()

# #--------------------------------------------plot generation & distance--------------------------------------------
# Load the data into a pandas DataFrame
df = pd.read_csv(path + "summary_"+f'{task}'+".csv")

# Calculate the p-value using Welch's t-test
Lamarckian_Learning = df[df['experiment']=='Lamarckian+Learning']
Darwinian_Learning = df[df['experiment']=='Darwinian+Learning']
_, p_value = ttest_ind(Lamarckian_Learning['distance'], Darwinian_Learning['distance'], equal_var=False)

plt.figure(figsize=(7, 6))
# Create a scatter plot with different colors for "Lamarckian+Learning" and "Darwinian+Learning"
ax = sns.lineplot(data=df, x='generation', y='distance', hue='experiment', palette=['mediumpurple', 'deepskyblue']) #palette='viridis'

ax.set_xlim(1, None)
ax.grid(True, alpha=0.2)

ax.set_title('Point Navigation (p-value: {:.2e})'.format(p_value), fontsize=12)
ax.set_xlabel('generation', fontsize=12)
ax.set_ylabel('distance',fontsize=12)

# Remove the legend title
ax.get_legend().set_title('')

plt.savefig(path+"plot_images/tree_dist_gen_"+f'{task}'+".png", bbox_inches='tight')
plt.show()


# # #--------------------------------------------plot dist & fitness--------------------------------------------
# df = pd.read_csv(path + "summary_"+f'{task}'+".csv")
#
# # Create a scatter plot with different colors for "Lamarckian+Learning" and "Darwinian+Learning"
# ax = sns.lineplot(data=df, x='distance', y='after', hue='experiment', palette=['mediumpurple', 'deepskyblue']) #palette='viridis'
#
# ax.set_xlim(1, None)
# ax.grid(True, alpha=0.3)
#
# ax.set_title('Point Navigation', fontsize=11)
# ax.set_xlabel('tree edit distance', fontsize=11)
# ax.set_ylabel('fitness',fontsize=11)
#
# # Rename the legend labels
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, ['Lamarckian+Learning','Darwinian+Learning'])
#
# plt.savefig(path+"plot_images/tree_dist_fitness_line_"+f'{task}'+".png", bbox_inches='tight')
# plt.show()
#
# # # #--------------------------------------------plot dist & learning delta--------------------------------------------
# df = pd.read_csv(path + "summary_"+f'{task}'+".csv")
#
# # Create a scatter plot with different colors for "Lamarckian+Learning" and "Darwinian+Learning"
# ax = sns.lineplot(data=df, x='distance', y='learning_delta', hue='experiment', palette=['mediumpurple', 'deepskyblue']) #palette='viridis'
#
# ax.set_xlim(1, None)
# ax.grid(True, alpha=0.3)
#
# ax.set_title('Point Navigation', fontsize=11)
# ax.set_xlabel('tree edit distance', fontsize=11)
# ax.set_ylabel('learning delta',fontsize=11)
#
# # Rename the legend labels
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, ['Lamarckian+Learning','Darwinian+Learning'])
#
# plt.savefig(path+"plot_images/tree_dist_delta_line_"+f'{task}'+".png", bbox_inches='tight')
# plt.show()