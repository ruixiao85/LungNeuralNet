args <- commandArgs(trailingOnly = TRUE)
wd <- ifelse(
   length(args) < 1,
   "./10x_scan_lung_smoke",
   is.na(args[1])
)

list.of.packages <- c("ggplot2", "data.table","reshape")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(ggplot2)
setwd(wd)
files <- dir(pattern = ".csv")
# filter <- "1.0_512"
# filter <- "softmax"
filter <- "UNet1a"
cfg<-'cat_resize_dim_net_filt_down_up_func'
df_list = NULL
for (f in files) {
   if (grepl(filter, f)) {
      print(f)
      df <- read.csv(f)
      df[,cfg] <- f
      df_list[[f]] <- df
      print(df[1:3,])
   }
}
print(length(df_list))
# print(file_df)

library(data.table)
head(df_all <- rbindlist(df_list))
df <- as.data.frame(df_all)

require(reshape)
df <- transform(df, config = colsplit(df[,cfg], split = "_", names = strsplit(cfg,"_")[[1]]))
# df <- data.frame(do.call('rbind', strsplit(as.character(df[,config]),'_',fixed=TRUE)))
writefile=paste0("all",filter,".csv")
write.csv(df,writefile)
print(writefile)
