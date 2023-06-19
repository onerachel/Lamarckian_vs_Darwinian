import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde, pearsonr

path = "/Users/lj/revolve2/NSR_data/"
task = "point_nav" #point_nav
df = pd.read_csv(path + "summary_"+f'{task}'+".csv")

grouped = df.groupby(['generation', 'experiment', 'individual_id']).agg({'after': 'mean', 'distance': 'mean', 'learning_delta': 'mean'}).reset_index()

filtered_grouped = grouped[grouped['generation'] != 0]

lamarckian_filtered = filtered_grouped[filtered_grouped['experiment'] == 'Lamarckian+Learning']
darwinian_filtered = filtered_grouped[filtered_grouped['experiment'] == 'Darwinian+Learning']

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

filtered_lamarckian_corr = df[(df['experiment'] == 'Lamarckian+Learning') & (df['generation'] != 0)][['after', 'distance']].corr().iloc[0, 1]
filtered_darwinian_corr = df[(df['experiment'] == 'Darwinian+Learning') & (df['generation'] != 0)][['after', 'distance']].corr().iloc[0, 1]

def plot_2d_density(ax, data, cmap):
    y = data['after']
    x = data['distance']

    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    density = kde(xy)

    xx, yy = np.mgrid[0:10:100j, 0:45:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kde(positions).T, xx.shape)

    img = ax.contourf(xx, yy, f, cmap=cmap)

    ax.set_xlabel('distance', fontsize=13)
    ax.set_ylabel('fitness', fontsize=13)
    ax.set_xlim([0, 10])
    # ax.set_ylim([0, 45])
    ax.set_ylim([0, 2.5])

    # Add grid lines
    ax.grid(True, linestyle='-', color='black', alpha=0.2)

    # Add correlation line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r", linewidth=1)

    return img, pearsonr(x, y)[0]

img1, corr1 = plot_2d_density(axs[0], lamarckian_filtered, 'Purples')
axs[0].set_title(f'Lamarckian+Learning (Corr: {filtered_lamarckian_corr:.2f})', fontsize=13)

img2, corr2 = plot_2d_density(axs[1], darwinian_filtered, 'Blues')
axs[1].set_title(f'Darwinian+Learning (Corr: {filtered_darwinian_corr:.2f})', fontsize=13)

# Create separate axes for the color bars
cax1 = fig.add_axes([0.47, 0.15, 0.02, 0.7])
fig.colorbar(img1, cax=cax1, shrink=0.5, aspect=5, ticks=[0, 0.1, 0.2, 0.3, 0.4])
# fig.colorbar(img1, cax=cax1, shrink=0.5, aspect=5, ticks=[0,0.008, 0.016,0.024])
cax2 = fig.add_axes([0.91, 0.15, 0.02, 0.7])
fig.colorbar(img2, cax=cax2, shrink=0.5, aspect=5, ticks=[0, 0.1, 0.2, 0.3])
# fig.colorbar(img2, cax=cax2, shrink=0.5, aspect=5, ticks=[0,0.008, 0.016,0.024])

plt.subplots_adjust(wspace=0.3)  # Adjust the space between subplots
fig.suptitle('Point Navigation', fontsize=14, fontweight='bold', x=0.05, y=0.5, ha='center', va='center', rotation='vertical')

plt.savefig(path+"plot_images/tree_dist_fitness_2d_density_subplots_"+f'{task}'+".png", bbox_inches='tight')
plt.show()
