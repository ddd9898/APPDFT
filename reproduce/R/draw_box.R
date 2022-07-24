setwd('~\\reproduce\\R')

##Code 1
data <- read.table(file="./boxdata_compare1.csv",header = TRUE, sep = ',')
head(data)

library(ggpubr)
library(patchwork)

library(ggplot2)
str(iris)


palettes = c("#8dd3c7","#fed9a6","#bebada","#80b1d3","#fb8072")

#AUC 
p1 = ggboxplot(data,x="Method",y="AUC.ROC",color="Method",palette=palettes,add="jitter",ylim=c(0,1))+
  stat_compare_means(comparisons = list(
    c("MHCFlurry","APPDFT"),
    c("MixMHCpred","APPDFT"),
    c("NetMHCpan4.0","APPDFT"),
    c("NetMHCpan4.1","APPDFT")),
    label.y = c(0.4,0.5,0.6,0.7))
p1
ggsave(filename = "./AUC.pdf",width=6,height = 4)
ggsave(filename = "./AUC.png",width=6,height = 4)

values = data[data$Method=='NetMHCpan4.1',c("AUC.ROC","AUC0.1.ROC","PPV")]
median(values[,1])
median(values[,2])
median(values[,3])

#AUC0.1
p1 = ggboxplot(data,x="Method",y="AUC0.1.ROC",color="Method",palette=palettes,add="jitter",ylim=c(0,1))+
  stat_compare_means(comparisons = list(
    c("MHCFlurry","APPDFT"),
    c("MixMHCpred","APPDFT"),
    c("NetMHCpan4.0","APPDFT"),
    c("NetMHCpan4.1","APPDFT")),
    label.y = c(0.1,0.2,0.3,0.4))
p1
ggsave(filename = "./AUC0.1.pdf",width=6,height = 4)
ggsave(filename = "./AUC0.1.png",width=6,height = 4)

#PPV 
p1 = ggboxplot(data,x="Method",y="PPV",color="Method",palette=palettes,add="jitter",ylim=c(0,1))+
  stat_compare_means(comparisons = list(
    c("MHCFlurry","APPDFT"),
    c("MixMHCpred","APPDFT"),
    c("NetMHCpan4.0","APPDFT"),
    c("NetMHCpan4.1","APPDFT")),
    label.y = c(0.1,0.2,0.3,0.4))
p1
ggsave(filename = "./PPV.pdf",width=6,height = 4)
ggsave(filename = "./PPV.png",width=6,height = 4)