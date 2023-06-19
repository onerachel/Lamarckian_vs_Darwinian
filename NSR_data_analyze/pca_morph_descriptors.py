import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import numpy as np

path = "/Users/lj/revolve2"
task = "point_nav"  # point_nav, rotation

# Read files
df = pd.read_csv(path + "/NSR_data/summary_" + f'{task}' + ".csv")

# Filter the DataFrame for the two experiment types
lamarckian_df = df[df['experiment'] == 'Lamarckian+Learning']
darwinian_df = df[df['experiment'] == 'Darwinian+Learning']

# Select the desired columns for the PCA analysis
columns = ['rel_size', 'proportion', 'rel_num_limbs', 'symmetry', 'branching',
           'coverage', 'rel_num_bricks', 'rel_num_hinges']

# Perform PCA separately for each experiment type
pca = PCA(n_components=2)

lamarckian_pca = pca.fit_transform(lamarckian_df[columns])
darwinian_pca = pca.fit_transform(darwinian_df[columns])

# Create a new figure and set the plot style
plt.figure(figsize=(8, 8))

# Plot Lamarckian+Learning
sns.scatterplot(x=lamarckian_pca[:, 0], y=lamarckian_pca[:, 1], s=10, marker='o', color='yellow', edgecolor='black', label='Lamarckian+Learning')

# Plot Darwinian+Learning
sns.scatterplot(x=darwinian_pca[:, 0], y=darwinian_pca[:, 1], s=10, marker='o', color='blue', edgecolor='black', label='Darwinian+Learning')

# Add labels for the variables
for i, column in enumerate(columns):
    plt.text(pca.components_[0, i], pca.components_[1, i], column, color='red', fontsize=10)

# Set x-axis and y-axis lines to pass through (0, 0)
plt.axhline(0, color='black', linewidth=0.5, linestyle='dashed')
plt.axvline(0, color='black', linewidth=0.5, linestyle='dashed')

# Set x-axis and y-axis ranges
plt.xlim(-1.5, 1.5)
plt.ylim(-1, 2)

# # Add confidence ellipses
# lamarckian_cov = np.cov(lamarckian_pca.T)
# darwinian_cov = np.cov(darwinian_pca.T)
#
# lamarckian_center = np.mean(lamarckian_pca, axis=0)
# darwinian_center = np.mean(darwinian_pca, axis=0)
#
# lamarckian_ellipse = mpatches.Ellipse(lamarckian_center, lamarckian_cov[0, 0], lamarckian_cov[1, 1], fill=False, edgecolor='red')
# darwinian_ellipse = mpatches.Ellipse(darwinian_center, darwinian_cov[0, 0], darwinian_cov[1, 1], fill=False, edgecolor='red')
#
# plt.gca().add_patch(lamarckian_ellipse)
# plt.gca().add_patch(darwinian_ellipse)

# Set plot title and legend
plt.title('PCA plot from 8-feature morphological traits')
plt.legend(title='')

# Show the plot
plt.grid(True, alpha=0.3)
plt.show()
