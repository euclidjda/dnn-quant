#!/usr/bin/env Rscript

# Load libraries
library(stringr)
library(scales)
library(ggplot2)
library(ggthemes)
library(RColorBrewer)
library(plyr)
library(tidyr)

# For this script to work we need the file output.dat

df <- read.table("output_data.dat",header=TRUE)

df$y0 <- floor(df$p0+0.49)
df$y1 <- floor(df$p1+0.49)

df$success <- df$t0*df$y0 + df$t1*df$y1

summary <- ddply(df,~date,summarise,mean=mean(success))

summary$date <- factor(summary$date)

g <- ggplot(summary,aes(x=date,y=mean,color=mean))

g <- g + geom_point()

g <- g + scale_x_discrete("Date",breaks = c("197001", "198001", "199001", "200001", "200801"))

g <- g + scale_y_continuous("Success Rate - 3 Years Out",limit=c(0.25,0.90))

g <- g + geom_hline(aes(yintercept=mean(summary$mean)))
