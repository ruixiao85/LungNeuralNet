print("Expected arguemnts:\n number of outputs (1)\n work_directory(current)")
args <- commandArgs(trailingOnly = TRUE)

if (is.na(args[1])) { nout=1 } else { nout=as.integer(args[1]) }
if (nout==1) { indicator='dice'; trend=max } else { indicator='acc'; trend=min }
if (!is.na(args[2])) { setwd(args[2]) }
if (is.na(args[3])) { prof="tissue_scale_reso_type_arch_param_conv_fun" } else { prof=args[3] }
name_profile=unlist(strsplit(prof,"_"))
nprofile=length(name_profile)

print(paste0("number of outputs: ",nout))
print(paste0("work directory: ",getwd()))
print(paste0("name of profiles: ",name_profile))

# setwd("D:/Cel files/Python/LungNeuralNet/10x_scan_lung_smoke")

# csv summary
#	val_loss	val_jac	val_dice	loss	jac	dice	lr	time	repeat
#0	0.174158634	0.654983198	0.769789769	0.24115942	0.527285925	0.649597633	1.00E-05	3/12/2019 20:28	1


require(reshape)
files=dir(pattern=paste0(nout,".csv")); fn=paste0("score_",nout,"_all")
dfa=NULL
# sink(paste0(fn,".csv"))
# print("Profile,Tissue,Repeat,Score\n");
for (f in files){
   df=read.csv(f,header=T,stringsAsFactors=F)
   elements=unlist(strsplit(f,"_"))
   if (length(elements)==nprofile) {
      df.long<-melt(df, id = c("time",'X'), measure = c(indicator,paste0("val_",indicator)))
      df<-aggregate(value~time+variable,data=df.long,FUN=trend) # melt long and taken best
      df$file=f
      for (e in 1:nprofile){ df[name_profile[e]]=elements[e] }
      if (is.null(dfa)) {dfa=df} else {dfa=rbind(dfa,df)}
   } else {
      print(paste("Skipping",f,"for incorrent number of profile elements",sep=' '))
   }
}
write.csv(dfa,paste0(fn,".csv"),row.names=F)


library(ggplot2)
pdf(paste0(fn,".pdf"),width=6.5,height=6)
ggplot(dfa,aes(x=conv,y=value,colour=variable,fill=variable))+geom_boxplot()+
   facet_grid(tissue~scale+reso+type+arch+param+fun,scales = "free_y")+
   theme_bw()+theme(axis.text.x = element_text(angle=90,hjust=1,vjust=0.5))
dev.off()


# p=ggplot(df3[df3$profile.mag=="1px/µm",], aes(x=Tissue,y=Score,group=profileall,color=profile.bone))+
#       geom_errorbar(aes(ymin=Score-Sd, ymax=Score+Sd),position=dodge,width=0.4)+
#       geom_line(aes(linetype=profile.mag),position=dodge)+
#       geom_point(aes(shape=profile.decon),alpha=0.8,position=dodge)+
#       facet_grid(.~profile.decon+profile.res+profile.mag)+
#       my_theme+ylim(0.4,1)


# sink()
#
# df2=read.csv(paste0(fn,".csv"),header=T,stringsAsFactors=F)
# df2m<-aggregate(.~Profile+Tissue,FUN=mean,data=df2[,-3])
# df2sd<-aggregate(.~Profile+Tissue,FUN=sd,data=df2[,-3])
# df2<-cbind(df2m,Sd=df2sd[,3])
#
# require(reshape)
# df3=transform(df2, profile=colsplit(Profile, split = "_", names = c('tis','mag','res','arch',
#                                                                     'bone','conv','decon','act')))
# df3$profile.bone=factor(df3$profile.bone,levels=c("Vgg16","V16T5","Res50","Dense121")) # order old->new
# df3$profile.mag=paste0(df3$profile.mag,"px/µm")
# df3$profile.mag=factor(df3$profile.mag,levels=c("1px/µm","0.8px/µm","0.6px/µm","0.4px/µm")) # order original->reduced
# df3$profile.decon=factor(substr(df3$profile.decon,1,4),levels=c("Ca3C","Ca33"))
# df3$profile.res=factor(df3$profile.res,levels=c("768x768","1024x1024"))
# df3$profileall=paste(df3$profile.mag,df3$profile.res,df3$profile.arch,
#                      df3$profile.bone,df3$profile.conv,df3$profile.decon,df3$profile.act,sep="_")
#
# my_theme<-theme_bw()+theme(axis.text.x = element_text(angle=90,hjust=1,vjust=0.5))
# dodge<-position_dodge(width=0.3)
# require("ggrepel") # text_repel
#
# pdf(paste0(fn,".pdf"),width=7,height=5,pointsize=12) # single
# p=ggplot(df3[df3$profile.mag=="1px/µm",], aes(x=Tissue,y=Score,group=profileall,color=profile.bone))+
#    geom_errorbar(aes(ymin=Score-Sd, ymax=Score+Sd),position=dodge,width=0.4)+
#    geom_line(aes(linetype=profile.mag),position=dodge)+
#    geom_point(aes(shape=profile.decon),alpha=0.8,position=dodge)+
#    facet_grid(.~profile.decon+profile.res+profile.mag)+
#    my_theme+ylim(0.4,1)
# p33<-ggplot(df3[df3$profile.decon=="Ca33"&df3$profile.res=="768x768",]
#             ,aes(x=Tissue,y=Score,group=profileall,color=profile.bone)) +
#    geom_errorbar(aes(ymin=Score-Sd, ymax=Score+Sd),position=dodge,width=0.4)+
#    geom_line(aes(linetype=profile.mag),position=dodge)+
#    geom_point(aes(shape=profile.decon),alpha=0.8,position=dodge)+
#    facet_grid(.~profile.decon+profile.res+profile.mag)+
#    my_theme+ylim(0.4,1)
#
# p
# p33
# p+geom_text_repel(aes(label=round(Score,2)),size=3,alpha=0.8,position=dodge)
# p33+geom_text_repel(aes(label=round(Score,2)),size=3,alpha=0.8,position=dodge)
#
# dev.off()
#
#
# pdf(paste0(fn,".pdf"),width=7,height=5,pointsize=12) # multi
# p=ggplot(df3[df3$profile.mag=="1px/µm",], aes(x=Tissue,y=Score,group=profileall,color=profile.bone))+
#    geom_errorbar(aes(ymin=Score-Sd, ymax=Score+Sd),position=dodge,width=0.4)+
#    geom_line(aes(linetype=profile.mag),position=dodge)+
#    geom_point(aes(shape=profile.decon),alpha=0.8,position=dodge)+
#    facet_grid(.~profile.act+profile.decon+profile.res+profile.mag)+
#    my_theme+ylim(0.8,1)
# p+geom_text_repel(aes(label=round(Score,2)),size=3,alpha=0.8,position=dodge)
#
# dev.off()
