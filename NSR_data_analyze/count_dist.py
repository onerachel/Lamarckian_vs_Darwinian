import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the CSV file
path = "/Users/lj/revolve2/NSR_data/"
task = "point_nav"  #rotation, point_nav
df = pd.read_csv(path + "best_parent_"+f'{task}'+"_distance.csv")

# Group the data by "run" and "dist", and count the number of unique "individual_id_x" values for each group
grouped_df = df.groupby(["run", "dist", "experiment"]).agg({"individual_id_x": pd.Series.nunique}).reset_index()

# Filter the data to only include the "Lamarckian+Learning" experiments
lamarckian_df = grouped_df[grouped_df["experiment"] == "Lamarckian+Learning"]

# Filter the data to only include the "Darwinian+Learning" experiments
darwinian_df = grouped_df[grouped_df["experiment"] == "Darwinian+Learning"]

# Calculate the mean and median of "dist" for each group
lamarckian_mean = lamarckian_df["dist"].mean()
lamarckian_median = lamarckian_df["dist"].median()

darwinian_mean = darwinian_df["dist"].mean()
darwinian_median = darwinian_df["dist"].median()

# Plot the data on separate subplots for the two experiments
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].bar(lamarckian_df["dist"], lamarckian_df["individual_id_x"])
axs[0].set_title(f"Lamarckian+Learning (Mean: {lamarckian_mean:.2f}, Median: {lamarckian_median:.2f})")
axs[0].set_xlabel("dist")
axs[0].set_ylabel("Count of individual_id_x")
axs[0].set_xlim(left=0)

axs[1].bar(darwinian_df["dist"], darwinian_df["individual_id_x"])
axs[1].set_title(f"Darwinian+Learning (Mean: {darwinian_mean:.2f}, Median: {darwinian_median:.2f})")
axs[1].set_xlabel("dist")
axs[1].set_ylabel("Count of individual_id_x")
axs[1].set_xlim(left=0)

plt.show()
