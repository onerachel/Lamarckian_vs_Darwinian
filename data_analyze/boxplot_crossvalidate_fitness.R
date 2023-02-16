# install.packages("/Users/lj/Downloads/ggpubr_0.5.0.tgz",repos=NULL)
# install.packages("/Users/lj/Downloads/rstatix_0.7.1.tgz",repos=NULL)
# install.packages("/Users/lj/Downloads/pbkrtest_0.5.2.tgz",repos=NULL)
# install.packages("/Users/lj/Downloads/car_3.1-1.tgz",repos=NULL)
library(ggplot2)
library(extrafont)
library(ggpubr)

df <- read.table("boxplot_3evaluations.csv",header=TRUE,sep=',') 
options( warn = -1 ) 
head(df)

df$nr_evaluations = factor(df$nr_evaluations)
df$framework= factor(df$framework)

# ggplot(data=df, aes(x=nr_evaluations, y=fitness, fill=framework)) + geom_boxplot() +
#   stat_summary(fun = mean,geom = "point", col = "red", 
#                position = position_dodge2(width = 0.75, preserve = "single"))

p0 =
    ggplot(data = df, aes(x=nr_evaluations, y=fitness, fill=framework)) +
    stat_boxplot(geom = "errorbar") + geom_boxplot() +
    stat_summary(fun = mean,geom = "point", col = "red", 
               position = position_dodge2(width = 0.75, preserve = "single")) +
    #ggtitle("best framework cross-validation") + 
    xlab("no. of evaluations") + ylab("fitness(cm/s)") +
  stat_compare_means(label = "p.signif")  +
    stat_compare_means(method = "anova" ,label.y = 35)
    


cols = rep(c("#9673A6","#B0E3E6","#6C8EBF"), length(levels(df$nr_evaluations)))
p=
p0 + theme_bw()+
      theme(axis.text.x=element_text(size=25,color = "grey20"), 
           axis.text.y=element_text(size=25,color = "grey20"),
           axis.title=element_text(size=25),
           plot.title = element_text(size=25, hjust = 0.5),
           legend.text=element_text(size=20),
           legend.title = element_text(size=20, hjust = 0.4),
           legend.key.height = unit(0.6,"cm"),
           legend.key.width = unit(0.6,"cm"), 
            legend.position = "top",
           text=element_text(family="Times New Roman", size=12)
           ) + 
  guides(fill=guide_legend(title="")) +
  scale_fill_manual(values=cols) 
  # + geom_dotplot(binaxis='y', stackdir='center', binwidth=1/170,
  #              position=position_dodge(1)) 
ggsave(p, filename = "framework_3evaluations_validation.pdf", device = cairo_pdf,height = 8, width = 8)

