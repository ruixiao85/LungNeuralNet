library(ggplot2)

setwd("D:/Cel files/Python/LungNeuralNet/10x_scan_lung_smoke")

files=dir(pattern="1.csv"); fn="score_1flat" # single
#files=dir(pattern="6.csv"); fn="score_6flat" # multi
#files=c(dir(pattern="1.csv"),dir(pattern="6.csv")); fn="score_1+6flat" # multi
sink(paste0(fn,".csv"))
cat("Profile,Tissue,Repeat,Score\n");
for (f in files){
   df=read.csv(f,header=T,stringsAsFactors=F)
   fc=gsub(",","",f)
   for (rep in c(1,2)){
      if (endsWith(f,"1.csv")) {
         cat(paste0(gsub("./","",f),",",strsplit(f,"_")[[1]][1],",",rep,",",
           max(df[df$'repeat.'==rep,]$'val_dice'),"\n")) # single
      } else {
        cat(paste0(gsub("./","",fc),",",strsplit(fc,"_")[[1]][1],",",rep,",",
           max(df[df$'repeat.'==rep,]$'val_acc'),"\n")) # multi
      }
   }
}
sink()

df2=read.csv(paste0(fn,".csv"),header=T,stringsAsFactors=F)
df2m<-aggregate(.~Profile+Tissue,FUN=mean,data=df2[,-3])
df2sd<-aggregate(.~Profile+Tissue,FUN=sd,data=df2[,-3])
df2<-cbind(df2m,Sd=df2sd[,3])

require(reshape)
df3=transform(df2, profile=colsplit(Profile, split = "_", names = c('tis','mag','res','arch',
                                                                    'bone','conv','decon','act')))
df3$profile.bone=factor(df3$profile.bone,levels=c("Vgg16","V16T5","Res50","Dense121")) # order old->new
df3$profile.mag=paste0(df3$profile.mag,"px/µm")
df3$profile.mag=factor(df3$profile.mag,levels=c("1px/µm","0.8px/µm","0.6px/µm","0.4px/µm")) # order original->reduced
df3$profile.decon=factor(substr(df3$profile.decon,1,4),levels=c("Ca3C","Ca33"))
df3$profile.res=factor(df3$profile.res,levels=c("768x768","1024x1024"))
df3$profileall=paste(df3$profile.mag,df3$profile.res,df3$profile.arch,
                     df3$profile.bone,df3$profile.conv,df3$profile.decon,df3$profile.act,sep="_")

my_theme<-theme_bw()+theme(axis.text.x = element_text(angle=90,hjust=1,vjust=0.5))
dodge<-position_dodge(width=0.3)
require("ggrepel") # text_repel

pdf(paste0(fn,".pdf"),width=7,height=5,pointsize=12) # single
p=ggplot(df3[df3$profile.mag=="1px/µm",], aes(x=Tissue,y=Score,group=profileall,color=profile.bone))+
   geom_errorbar(aes(ymin=Score-Sd, ymax=Score+Sd),position=dodge,width=0.4)+
   geom_line(aes(linetype=profile.mag),position=dodge)+
   geom_point(aes(shape=profile.decon),alpha=0.8,position=dodge)+
   facet_grid(.~profile.decon+profile.res+profile.mag)+
   my_theme+ylim(0.4,1)
p33<-ggplot(df3[df3$profile.decon=="Ca33"&df3$profile.res=="768x768",]
            ,aes(x=Tissue,y=Score,group=profileall,color=profile.bone)) +
   geom_errorbar(aes(ymin=Score-Sd, ymax=Score+Sd),position=dodge,width=0.4)+
   geom_line(aes(linetype=profile.mag),position=dodge)+
   geom_point(aes(shape=profile.decon),alpha=0.8,position=dodge)+
   facet_grid(.~profile.decon+profile.res+profile.mag)+
   my_theme+ylim(0.4,1)
   
p
p33
p+geom_text_repel(aes(label=round(Score,2)),size=3,alpha=0.8,position=dodge)
p33+geom_text_repel(aes(label=round(Score,2)),size=3,alpha=0.8,position=dodge)

dev.off()


pdf(paste0(fn,".pdf"),width=7,height=5,pointsize=12) # multi
p=ggplot(df3[df3$profile.mag=="1px/µm",], aes(x=Tissue,y=Score,group=profileall,color=profile.bone))+
   geom_errorbar(aes(ymin=Score-Sd, ymax=Score+Sd),position=dodge,width=0.4)+
   geom_line(aes(linetype=profile.mag),position=dodge)+
   geom_point(aes(shape=profile.decon),alpha=0.8,position=dodge)+
   facet_grid(.~profile.act+profile.decon+profile.res+profile.mag)+
   my_theme+ylim(0.8,1)
p+geom_text_repel(aes(label=round(Score,2)),size=3,alpha=0.8,position=dodge)

dev.off()
