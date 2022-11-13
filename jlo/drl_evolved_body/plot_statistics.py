import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('./statistics.csv')

fig = plt.figure(figsize=(13, 7))
ax = fig.add_subplot(1, 2, 1)
ax.title.set_text("Mean reward for each action")
plt.plot(data['mean_rew'])
ax = fig.add_subplot(1, 2, 2)
ax.title.set_text("Mean value for each state")
plt.plot(data['mean_val'])
plt.show()