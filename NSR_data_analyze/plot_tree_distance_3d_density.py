import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D

path = "/Users/lj/revolve2/NSR_data/"
task = "point_nav"
df = pd.read_csv(path + "summary_"+f'{task}'+".csv")

grouped = df.groupby(['generation', 'experiment', 'individual_id']).agg({'after': 'mean', 'distance': 'mean', 'learning_delta': 'mean'}).reset_index()

filtered_grouped = grouped[grouped['generation'] != 0]

lamarckian_filtered = filtered_grouped[filtered_grouped['experiment'] == 'Lamarckian+Learning']
darwinian_filtered = filtered_grouped[filtered_grouped['experiment'] == 'Darwinian+Learning']

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

def plot_3d_density(ax, data, cmap):
    z = data['distance']
    y = data['after']
    x = data['generation']

    xyz = np.vstack([x, y, z])
    kde = gaussian_kde(xyz)
    density = kde(xyz)

    try:
        idx = density.argsort()
        x, y, z, density = x.iloc[idx], y.iloc[idx], z.iloc[idx], density[idx]
    except KeyError as e:
        print(f"KeyError: {e}")

    img = ax.scatter(x, y, z, c=density, cmap=cmap, s=10)
    fig.colorbar(img, ax=ax, shrink=0.5, aspect=5, label='Density')

    ax.set_xlabel('Distance')
    ax.set_ylabel('Fitness')
    ax.set_zlabel('Learning Delta')

plot_3d_density(ax1, lamarckian_filtered, 'viridis')
ax1.set_title('Lamarckian+Learning')

plot_3d_density(ax2, darwinian_filtered, 'viridis')
ax2.set_title('Darwinian+Learning')

plt.savefig(path+"plot_images/tree_dist_fitness_3d_density_subplots_"+f'{task}'+".png", bbox_inches='tight')
plt.show()
