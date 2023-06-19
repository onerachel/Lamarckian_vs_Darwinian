import pandas as pd
import numpy as np

path = "/Users/lj/revolve2"
task = "point_nav"  # point_nav, rotation

# Read files
df = pd.read_csv(path + "/NSR_data/summary_" + f'{task}' + ".csv")

# Define a function to calculate the mean, standard error, and largest values
def calculate_stats(column):
    mean = column.mean()
    std_error = column.sem()
    largest = column.nlargest(3).mean()
    return mean, std_error, largest

# Iterate over the feature columns and calculate the statistics
for feature in ['rel_size', 'proportion', 'rel_num_limbs', 'symmetry', 'branching',
                'coverage', 'rel_num_bricks', 'rel_num_hinges']:
    mean, std_error, largest = calculate_stats(df[feature])
    df[f'{feature}_mean'] = mean
    df[f'{feature}_std_error'] = std_error
    df[f'{feature}_largest'] = largest

# Save the updated DataFrame to the CSV file
df.to_csv(path + "/NSR_data/pca_" + f'{task}' + ".csv")
