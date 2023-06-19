import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

# Load the CSV file
path = "/Users/lj/revolve2/NSR_data/"
task = "point_nav"  #rotation, point_nav
df = pd.read_csv(path + "best_parent_"+f'{task}'+"_distance.csv")

# Define a function to calculate the number of unique bodies based on the 7 morphological descriptors
def count_unique_bodies(group):
    return len(group.groupby(['rel_num_hinges_x', 'rel_num_limbs_x', 'proportion_x', 'rel_num_bricks_x', 'symmetry_x', 'branching_x','coverage_x']))

# Calculate the number of unique bodies for each generation and experiment
generation_diff_bodies = df.groupby(['experiment', 'generation_x', 'run']).apply(count_unique_bodies).reset_index(name='num_diff_bodies')
generation_diff_bodies['diversity'] = generation_diff_bodies['num_diff_bodies'] / 1624

# Define the colors
colors = ['deepskyblue', 'mediumpurple']
fill_colors = ['lightskyblue', 'mediumpurple']

# Calculate mean and confidence interval
mean_ci_diff_bodies = generation_diff_bodies.groupby(['experiment', 'generation_x']).agg({'diversity': ['mean', 'std', 'count']}).reset_index()
mean_ci_diff_bodies['ci_low'] = mean_ci_diff_bodies['diversity']['mean'] - 1.96 * (mean_ci_diff_bodies['diversity']['std'] / np.sqrt(mean_ci_diff_bodies['diversity']['count']))
mean_ci_diff_bodies['ci_high'] = mean_ci_diff_bodies['diversity']['mean'] + 1.96 * (mean_ci_diff_bodies['diversity']['std'] / np.sqrt(mean_ci_diff_bodies['diversity']['count']))

# Perform t-test and calculate p-value
experiment_0_data = generation_diff_bodies[generation_diff_bodies['experiment'] == 0]['diversity']
experiment_1_data = generation_diff_bodies[generation_diff_bodies['experiment'] == 1]['diversity']
t_statistic, p_value = ttest_ind(experiment_0_data, experiment_1_data)

# Plot the line plot with mean and confidence interval
ax = sns.lineplot(x="generation_x", y="diversity", hue="experiment", data=generation_diff_bodies, palette=colors)
for i, color in enumerate(fill_colors):
    ax.fill_between(mean_ci_diff_bodies[mean_ci_diff_bodies['experiment'] == i]['generation_x'], mean_ci_diff_bodies[mean_ci_diff_bodies['experiment'] == i]['ci_low'], mean_ci_diff_bodies[mean_ci_diff_bodies['experiment'] == i]['ci_high'], color=color, alpha=0.3)

# Set the title and axis labels
title = f"Point Navigation (p-value={p_value:.4f})"
ax.set_title(title, fontsize=11)
ax.set_xlabel('generation')
ax.set_ylabel('diversity')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['Darwinian+Learning', 'Lamarckian+Learning'])
plt.savefig(path+"plot_images/diversity_"+f'{task}'+"2.png", bbox_inches='tight')
plt.show()

# # Plot the line plot with mean and standard deviation
# ax = sns.lineplot(x="generation_x", y="num_diff_bodies", hue="experiment", ci='sd', data=generation_diff_bodies)
# # Set the title and axis labels
# ax.set_title('Rotation', fontsize=11)
# ax.set_xlabel('generation')
# ax.set_ylabel('num of different bodies')
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, ['Darwinian+Learning','Lamarckian+Learning'])
# plt.savefig(path+"plot_images/num_diff_bodies_"+f'{task}'+".png", bbox_inches='tight')
# plt.show()