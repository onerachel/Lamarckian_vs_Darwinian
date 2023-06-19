library(ggplot2)
library(dplyr)



# read in the CSV file
df <- read.table("best_parent_point_nav_distance.csv",header=TRUE,sep=',')  

#df <- filter(df, run == 3)

# Create separate histograms for each value of the "experiment" variable
p <- ggplot(df, aes(x = dist, fill = experiment)) + theme_bw()+
  geom_histogram(alpha = 1, bins = 20) +
  facet_wrap(~experiment) +
  ggtitle(paste("Point Navigation", sprintf("(p = %0.2f)", t.test(df[df$experiment == "Lamarckian+Learning", "dist"],
                                                          df[df$experiment == "Darwinian+Learning", "dist"])$p.value))) +
  labs(x = "distance", y = "count", fill = "experiment") +
  scale_fill_manual(values = c("#6C8EBF", "#9673A6")) + # change colors
  ylim(0, 11500) +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, size = 11, face="bold")) # hide the legend

# Calculate the average "dist" value for each "experiment" value
avg_dist <- df %>%
  group_by(experiment) %>%
  summarise(mean_dist = mean(dist))

# Add the average "dist" values to the plot as a point
p <- p +
  geom_point(data = avg_dist, aes(x = mean_dist, y = 0),
             color = "red", size = 1)+
  geom_text(data = avg_dist, aes(x = mean_dist, y = 0, label = sprintf("%.2f", mean_dist)),
            vjust = -1)

print(p)
ggsave(p, filename = "boxplot_dist_point_nav_runs.png",height = 8, width = 8)
