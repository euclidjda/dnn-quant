library(ggplot2)
library(tidyr)
library(ggthemes)

df <- read.table("~/work/research/research-cetox/source-data/train-3yr.dat",header=TRUE)

df.wide <- df[9:47]

df.wide.norm <- df.wide[ c(1,grep("norm",names(df.wide))) ]
df.long.norm <- gather(df.wide.norm,feature,value,2:ncol(df.wide.norm))
norm.samp    <- df.long.norm[sample(1:nrow(df.long.norm),100000,replace=FALSE),]

df.wide.rel  <- df.wide[ c(1,grep("rel",names(df.wide))) ]
df.long.rel  <- gather(df.wide.rel,feature,value,2:ncol(df.wide.rel))
lchg.samp    <- df.long.lchg[sample(1:nrow(df.long.lchg),100000,replace=FALSE),]

df.wide.lchg  <- df.wide[ c(1,grep("lchg",names(df.wide))) ]
df.long.lchg  <- gather(df.wide.lchg,feature,value,2:ncol(df.wide.lchg))
rel.samp      <- df.long.rel[sample(1:nrow(df.long.rel),100000,replace=FALSE),]

p <- ggplot( norm.samp, 
             aes( x=value, fill=target))+
             geom_density(alpha=0.4,position="identity")+
             facet_wrap( ~feature, scales='free' )+theme_bw()

p <- ggplot( lchg.samp, 
             aes( x=value, fill=target))+
             geom_density(alpha=0.4,position="identity")+
             facet_wrap( ~feature, scales='free' )+theme_bw()

p <- ggplot( rel.samp, 
             aes( x=value, fill=target))+
             geom_density(alpha=0.4,position="identity")+
             facet_wrap( ~feature, scales='free' )+theme_bw()


