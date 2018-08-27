args <- commandArgs(trailingOnly = TRUE)
wd <- ifelse(
   length(args) < 1,
   "D:/Cel files/2018-07.08 Keras Tensorflow 2X IMAGES/LungNeuralNet/10x_scan_lung_smoke",
   is.na(args[1])
)

library(ggplot2)
setwd(wd)
files <- dir(pattern = ".csv")
# filter <- "1.0_512"
# filter <- "softmax"
filter <- "unet1"
config<-'cat_resize_dim_net_nout_filt_downconv_downsamp_upconv_upsamp_actf_outf_lossf'
df_list = NULL
for (f in files) {
   if (grepl(filter, f)) {
      # print(f)
      df <- read.csv(f)
      df[,config] <- f
      df_list[[f]] <- df
      # print(df[1:3,])
   }
}
print(length(df_list))
# print(file_df)

library(data.table)
head(df_all <- rbindlist(df_list))
df <- as.data.frame(df_all)

require(reshape)
df <- transform(df, config = colsplit(df[,config], split = "_",names = strsplit(config,"_")[[1]]))
# df <- data.frame(do.call('rbind', strsplit(as.character(df[,config]),'_',fixed=TRUE)))
# write.csv(df,paste0("all",filter,".csv"))


