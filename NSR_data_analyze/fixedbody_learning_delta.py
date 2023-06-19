import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

path = "/Users/lj/revolve2/NSR_data/"
# Read the CSV file into a pandas DataFrame
df = pd.read_csv(path + "fixedbody_learning_delta.csv")

# Group the data by 'gen_num', 'robot', and calculate the difference between the maximum and minimum 'fitness(cm/s)'
df['learning_delta'] = df.groupby(['gen_num', 'robot'])['fitness(cm/s)'].transform(lambda x: x.max() - x.min())

# Sort the DataFrame by 'gen_num'
df = df.sort_values('gen_num')

# Save the new data as a CSV file
df.to_csv(path + "fixedbody_learning_delta_new.csv", index=False)

# Get unique values in the 'robot' column
robots = df['robot'].unique()

# Create a line plot for each robot
for robot in robots:
    robot_df = df[df['robot'] == robot]  # Filter data for the specific robot
    plt.plot(robot_df['gen_num'], robot_df['learning_delta'], label=robot)

plt.xlabel('generation')
plt.ylabel('learning delta')
plt.title('Learning Delta over Generations')

# Format x-axis ticks as integers
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Set the x-axis limits from 1 to 20
plt.xlim(0, 20)

# Adjust the position of the legend
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0.)

plt.savefig('/Users/lj/revolve2/NSR_data/plot_images/fixedbody_learning_delta_individuals.png', bbox_inches='tight')
plt.show()
