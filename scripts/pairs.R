require(GGally)

df <- read.table("~/work/research/research-rnn/train.dat",header=TRUE)

df.nona <- subset(df, isna.rel.ebit_entval==0 & isna.rel.book_market==0)

df.nona$target <- factor(df.nona$target)

df.sub <- df.nona[sample(1:nrow(df.nona),1000,replace=FALSE),]

ggpairs(df.sub,columns=c('rel.ebit_entval','rel.book_market','rel.mom6m'), mapping = aes(color = target))
