import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde, pearsonr

path = "/Users/lj/revolve2/NSR_data/"
task = "point_nav"
df = pd.read_csv(path + "summary_"+f'{task}'+".csv")

grouped = df.groupby(['generation', 'experiment', 'individual_id']).agg({'after': 'mean', 'distance': 'mean', 'learning_delta': 'mean'}).reset_index()

filtered_grouped = grouped[grouped['generation'] != 0]

lamarckian_filtered = filtered_grouped[filtered_grouped['experiment'] == 'Lamarckian+Learning']
darwinian_filtered = filtered_grouped[filtered_grouped['experiment'] == 'Darwinian+Learning']

fig, ax = plt.subplots(figsize=(10, 8))

def plot_2d_density(ax, data, cmap, alpha, line_color):
    y = data['after']
    x = data['distance']

    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    density = kde(xy)

    xx, yy = np.mgrid[0:10:100j, 0:2.5:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kde(positions).T, xx.shape)

    img = ax.contourf(xx, yy, f, cmap=cmap, alpha=alpha)

    ax.set_xlabel('distance', fontsize=13)
    ax.set_ylabel('fitness', fontsize=13)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 2.5])

    # Add grid lines
    ax.grid(True, linestyle='-', color='black', alpha=0.2)

    # Add correlation line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), line_color, linewidth=1)

    return img, pearsonr(x, y)[0]

img1, corr1 = plot_2d_density(ax, lamarckian_filtered, 'Purples', 1.0, 'purple')
img2, corr2 = plot_2d_density(ax, darwinian_filtered, 'Blues', 0.5, 'darkblue')

ax.set_title(f'Point Navigation (Lamarckian r={corr1:.2f}), Darwinian r={corr2:.2f}))', fontsize=15)

# Create a separate axis for the color bars
cax1 = fig.add_axes([0.92, 0.55, 0.02, 0.3])
fig.colorbar(img1, cax=cax1, shrink=0.5, aspect=5, ticks=[0, 0.1, 0.2, 0.3, 0.4], label='Lamarckian+Learning')
cax2 = fig.add_axes([0.92, 0.15, 0.02, 0.3])
fig.colorbar(img2, cax=cax2, shrink=0.5, aspect=5,ticks=[0, 0.1, 0.2, 0.3, 0.4], label='Darwinian+Learning')

plt.savefig(path+"plot_images/tree_dist_fitness_2d_density_merged_"+f'{task}'+".png", bbox_inches='tight')
plt.show()
