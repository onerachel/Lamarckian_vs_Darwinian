# install.packages("/Users/lj/Downloads/ggpubr_0.5.0.tgz",repos=NULL)
# install.packages("/Users/lj/Downloads/rstatix_0.7.1.tgz",repos=NULL)
# install.packages("/Users/lj/Downloads/pbkrtest_0.5.2.tgz",repos=NULL)
# install.packages("/Users/lj/Downloads/car_3.1-1.tgz",repos=NULL)
library(ggplot2)
library(extrafont)
library(ggpubr)

df <- read.table("boxplot_final_gen_point_nav.csv",header=TRUE,sep=',') 
options( warn = -1 ) 
head(df)

df$generation = factor(df$generation)
df$experiment= factor(df$experiment)

# ggplot(data=df, aes(x=generation, y=fitness, fill=experiment)) + geom_boxplot() +
#   stat_summary(fun = mean,geom = "point", col = "red", 
#                position = position_dodge2(width = 0.75, preserve = "single"))

p0 =
    ggplot(data = df, aes(x=generation, y=fitness, fill=experiment)) +
    stat_boxplot(geom = "errorbar") + geom_boxplot() +
    stat_summary(fun = mean,geom = "point", size = 0.5, col = "red", 
               position = position_dodge2(width = 0.75, preserve = "single")) +
    #ggtitle("best experiment cross-validation") + 
    xlab("no. of generation") + ylab("fitness") +
    stat_compare_means(label = "p.signif")  +
    stat_compare_means(method = "anova" ,label.y = 2.7)
    

# "#B0E3E6", green

cols = rep(c("#6C8EBF","#9673A6"), length(levels(df$generation)))
p=
p0 + theme_bw()+
      theme(axis.text.x=element_text(size=25,color = "grey20"), 
           axis.text.y=element_text(size=25,color = "grey20"),
           axis.title=element_text(size=25),
           plot.title = element_text(size=25, hjust = 0.5),
           legend.text=element_text(size=15),
           legend.title = element_text(size=20, hjust = 0.4),
           legend.key.height = unit(0.6,"cm"),
           legend.key.width = unit(0.6,"cm"), 
            legend.position = "top",
           text=element_text(family="Times New Roman", size=10)
           ) + 
  guides(fill=guide_legend(title="Point Navigation Task")) +
  scale_fill_manual(values=cols) 
  # + geom_dotplot(binaxis='y', stackdir='center', binwidth=1/170,
  #              position=position_dodge(1)) 
# ggsave(p, filename = "boxplot_final_gen_rotation.pdf", device = cairo_pdf,height = 8, width = 8)
ggsave(p, filename = "boxplot_final_gen_point_nav.png",height = 8, width = 8)

