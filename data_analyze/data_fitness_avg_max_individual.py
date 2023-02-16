#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 4 18:47:58 2023

@author: LJ
"""

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import scipy.stats as st

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)
path = "/Users/lj/revolve2-Alife"

# Read files
df = pd.read_csv(path + "/databases_eval1000/data_analysis_1000_20runs.csv")

## max_max
# df=df.groupby(["robot"])["fitness(cm/s)"].agg(lambda x: x.max())

## avg_max(MBF)
# df=(df.groupby(["robot",'run'])["fitness(cm/s)"].agg(lambda x: x.max())).groupby(["robot"]).mean()

# avg_avg
# df=(df.groupby(["robot",'run'])["fitness(cm/s)"].agg(lambda x: x.mean())).groupby(["robot"]).mean()

## max_max_framework
# df = df.groupby(["robot", "controller+learner"])["fitness(cm/s)"].agg(
#     lambda x: x.max()).groupby(["controller+learner"])

## avg_max(MBF) framework
df=(df.groupby(["robot",'run', "controller+learner"])["fitness(cm/s)"].agg(lambda x: x.max())).groupby(["robot","controller+learner"]).mean()

## avg_avg framework
# df=(df.groupby(["robot",'run',"controller+learner"])["fitness(cm/s)"].agg(lambda x: x.mean())).groupby(["robot","controller+learner"]).mean()


# df=df.apply(pd.DataFrame)
print(df.head(200))
# df.to_csv(path + "/databases_eval1000/data_analysis_1000_max_max.csv")



