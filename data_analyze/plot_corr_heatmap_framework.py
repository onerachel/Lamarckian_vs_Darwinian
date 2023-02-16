import os
path = "/Users/lj/revolve2-Alife"
learner = "PPO"
# for dirname, _, filenames in os.walk(path+'/databases_eval580'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
sns.set(rc={"figure.figsize": (20, 15)})
sns.set_style("whitegrid")


# Read files
df_raw = pd.read_csv(path+"/databases_eval580/robotzoo_descriptors.csv")
pd.set_option('display.max_columns', None)
# print(df_raw.head())

df = df_raw[['avg_fitness', 'limbs', 'extremities', 'length_of_limbs', 'extensiveness',
                       'coverage', 'joints', 'hinge_count', 'active_hinges_count', 'brick_count',
                       'proportion', 'width', 'height', 'size', 'absolute_size', 'symmetry', 'max_fitness']]
# Heatmap for all the numerical data including the target 'fitness'
# Define the heatmap parameters
pd.options.display.float_format = "{:,.2f}".format

# Define correlation matrix
corr_matrix = df.corr()

# Replace correlation < |0.3| by 0 for a better visibility
# corr_matrix[(corr_matrix < 0.3) & (corr_matrix > -0.3)] = 0
cols = corr_matrix.nlargest(10, 'max_fitness')['max_fitness'].index
# plot the heatmap
sns.heatmap(corr_matrix, vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot_kws={"size": 9, "color": "black"},annot=True)
plt.title("Fitness Correlation")
plt.savefig(path+"/databases_eval580/plot_images/heat_map_framework.png")
plt.show()

## Lets visualize individually

# corr = df.corr()["avg_fitness"].sort_values(ascending=False)[1:7] ## selecting 6 cols other than fitness(cm/s)
# print(corr)
#
# f, ax = plt.subplots(nrows=6, ncols=1, figsize=(20, 40))
# for i, col in enumerate(corr.index):
#     sns.scatterplot(x=col, y="avg_fitness", data=df, ax=ax[i], color='darkorange')
#     ax[i].set_title(f'{col} vs fitness')
# # plt.savefig(path + "/databases_eval580/correlation_scatter_plot_"+learner+".png")
# plt.show()