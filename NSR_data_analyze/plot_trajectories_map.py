import numpy as np
from matplotlib import pyplot as plt
import asyncio
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd

async def main() -> None:

    fig, ax = plt.subplots()

    path = "/Users/lj/revolve2"
    trajectories = np.load(path+'/darwinian_data/sexual_traj.npy')

    trajectories = np.transpose(trajectories, axes=(1,0,2))
    trajectories = trajectories.reshape(2, -1)
    
    df = pd.DataFrame({'x':trajectories[0], 'y':trajectories[1]})
    sns.kdeplot(df, x='x', y='y', fill=True, cmap='viridis', cbar=True)

    target1 = plt.Circle((1,-1), 0.08, color='red')
    target2 = plt.Circle((0,-2), 0.08, color='red')
    starting_point = plt.Circle((0,0), 0.08, color='black')
    ax.add_patch(target1)
    ax.add_patch(target2)
    ax.add_patch(starting_point)

    green_circle = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                            markersize=10, label='Target point')
    black_circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                            markersize=10, label='Starting point')    
    ax.legend(handles=[green_circle, black_circle])
    plt.title('Asexual Reproduction')
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())