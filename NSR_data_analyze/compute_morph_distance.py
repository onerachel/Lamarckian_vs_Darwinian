#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 23 15:40:58 2023

@author: LJ
"""
from scipy.spatial import KDTree
import numpy as np
import pandas as pd


path = "/Users/lj/revolve2/NSR_data/"
task = "rotation"  # point_nav, rotation
pd.set_option('display.max_columns', 100)

# load the data
df1 = pd.read_csv(path + "summary_" + f'{task}' + ".csv")
df2 = pd.read_csv(path + f'{task}' + "_parent.csv")

# filter out child-parent pairs where the parent has a lower fitness value
df_two_parent = pd.merge(df2, df1, on=['run', 'individual_id', 'experiment'], how='left')
df_two_parent = df_two_parent.drop_duplicates(subset=['run', 'individual_id', 'experiment', 'parent_id'])
df_two_parent.to_csv(path + "two_parent_" + f'{task}' + ".csv", index=False)

df_one_parent = df_two_parent.loc[df_two_parent.groupby(['individual_id', 'run', 'experiment'])['after'].idxmax()]
df_final = pd.merge(df_one_parent, df1, left_on=['run', 'parent_id', 'experiment'],
                    right_on=['run', 'individual_id', 'experiment'], how='left')
df_final = df_final.drop_duplicates(subset=['run', 'individual_id_x', 'experiment', 'parent_id'])
df_final.to_csv(path + "best_parent_" + f'{task}' + ".csv", index=False)

# calculate the similarity(distance) between the child and the best parent
df_final['dist'] = np.sqrt((df_final['rel_num_hinges_x'] - df_final['rel_num_hinges_y']) ** 2 + (
            df_final['proportion_x'] - df_final['proportion_y']) ** 2 + (
                                       df_final['rel_num_bricks_x'] - df_final['rel_num_bricks_y']) ** 2
                           + (df_final['rel_num_limbs_x'] - df_final['rel_num_limbs_y']) ** 2 + (
                                       df_final['symmetry_x'] - df_final['symmetry_y']) ** 2 + (
                                       df_final['branching_x'] - df_final['branching_y']) ** 2 + (
                                       df_final['coverage_x'] - df_final['coverage_y']) ** 2)

df_final.to_csv(path + "best_parent_" + f'{task}' + "_distance.csv", index=False)