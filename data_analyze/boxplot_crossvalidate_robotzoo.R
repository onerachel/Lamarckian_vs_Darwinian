library(ggplot2)
library(extrafont)

df <- read.table("Swarm14_best_crossvalidate2.csv",header=TRUE,sep=',') 
options( warn = -1 ) 
head(df)

df$arena = factor(df$arena)
df$controller = factor(df$controller)

# ggplot(data=df, aes(x=arena, y=fitness, fill=controller)) + geom_boxplot() +
#   stat_summary(fun = mean,geom = "point", col = "red", 
#                position = position_dodge2(width = 0.75, preserve = "single"))




p0 =
  ggplot(data = df, aes(x=arena, y=fitness, fill=controller)) +
    stat_boxplot(geom = "errorbar") + geom_boxplot() +
    stat_summary(fun = mean,geom = "point", col = "red", 
               position = position_dodge2(width = 0.75, preserve = "single")) +
    #ggtitle("best controller cross-validation") + 
    xlab(NULL) + ylab("fitness")


cols = rep(c("#9673A6","#6C8EBF","#B0E3E6"), length(levels(df$arena)))
p=
p0 + theme_bw()+
      theme(axis.text.x=element_text(size=25,color = "grey20", angle = 10),
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
  guides(fill=guide_legend(title="controller")) +
  scale_fill_manual(values=cols) 
  # + geom_dotplot(binaxis='y', stackdir='center', binwidth=1/170,
  #              position=position_dodge(1)) 
ggsave(p, filename = "best_controller_cross_validation_new.pdf", device = cairo_pdf,height = 8, width = 8)

