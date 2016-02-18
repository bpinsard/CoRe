library(nlme)
library(lme4)
library(ggplot2)
library(lattice)
data<-read.csv('~/data/analysis/core_behav/test_r_file.csv')
data$transition<-as.factor(data$transition)

xyplot(rt_pre~correct_seq_idx|subject,
       group=~task,
       data=data[(data$transition=='23')&data$match&(data$sequence=='CoReTSeq'),])


##### LME4 #################

model1<-lme4::nlmer(
    rt_pre~SSasymp(correct_seq_idx/max(correct_seq_idx),Asym,xmid,scale)~ (Asym+xmid+scale | subject_id),
    start=c(Asym=.2,xmid=.5,scale=.1),
    data=data[(data$transition=='14')&data$match&(data$sequence=='CoReTSeq')&(data$group!='NoReactNoInt')&(data$group!='NoReactInt'),])
summary(model1)


cons_func2 <- deriv(
  ~learn_asym+(learn_start-learn_asym)*exp(-exp(learn_rate)*seq_id)+sleep_gain*rt1,
               namevec=c("learn_asym","learn_start","learn_rate","sleep_gain"),
               function.arg=c("learn_asym","learn_start","learn_rate","sleep_gain","seq_id","rt1"))
cons_res_nlmer<-lme4::nlmer(
  rt_pre~cons_func2(learn_asym, learn_start, learn_rate, sleep_gain, correct_seq_idx,task=='Reactivation-TSeq-D-Two')~ sleep_gain+(learn_asym+learn_start+learn_rate |subject_id),
  start=c(learn_asym=.2,learn_start=.5,learn_rate=1,sleep_gain=0),
  data=data[(data$transition=='14')&data$match&(data$sequence=='CoReTSeq')&(data$group!='NoReactNoInt')&(data$group!='NoReactInt')&(data$task!='Testing-TSeq-D-Three')&(data$rt_pre<2),],
)

##### NLME #################

cons_func <- function(learn_asym, learn_start, learn_rate, sleep_gain, task, seq_id) learn_asym+(learn_start-learn_asym)*exp(-exp(learn_rate)*seq_id)+sleep_gain*(task=='Reactivation-TSeq-D-Two') 
cons_model<-rt_pre~cons_func(learn_asym, learn_start, learn_rate, sleep_gain, task, correct_seq_idx)
summary(cons_res<-nlme::nlme(
    model=cons_model,
    fixed=learn_asym+learn_start+learn_rate+sleep_gain~1,
    random=learn_asym+learn_start+learn_rate~1,
    start=c(learn_asym=.2,learn_start=.5,learn_rate=1,sleep_gain=0),
    data=data[(data$transition=='23')&data$match&(data$sequence=='CoReTSeq')&(data$group!='NoReactNoInt')&(data$group!='NoReactInt')&(data$task!='Testing-TSeq-D-Three')&(data$rt_pre<2),],
    group=~subject_id))

transition_mask=(data$transition=='14')|(data$transition=='42')|(data$transition=='23')|(data$transition=='31')|(data$transition=='11')

summary(cons_res<-nlme::nlme(
  model=cons_model,
  fixed=learn_asym+learn_start+learn_rate+sleep_gain~1,
  random=learn_asym+learn_start+learn_rate~1,
  start=c(learn_asym=.2,learn_start=.5,learn_rate=1,sleep_gain=0),
  data=data[transition_mask&data$match&(data$sequence=='CoReTSeq')&(data$group!='NoReactNoInt')&(data$group!='NoReactInt')&(data$task!='Testing-TSeq-D-Three')&(data$rt_pre<2),],
  group=~subject_id))

interf_func <- function(learn_asym, learn_start, learn_rate, sleep_gain, interf, task, seq_id) learn_asym+(learn_start-learn_asym)*exp(-exp(learn_rate)*seq_id)+sleep_gain*(task!='Training-TSeq-D_One')+interf*(task=='Testing-TSeq-D-Three')
interf_model<-rt_pre~interf_func(learn_asym, learn_start, learn_rate, sleep_gain, interf, task, correct_seq_idx)

summary(interf_res<-nlme::nlme(
  model=interf_model,
  fixed=list(learn_asym+learn_start+learn_rate+sleep_gain+interf~1),
  random=learn_asym+learn_start+learn_rate~1,
  start=c(learn_asym=.2,learn_start=.5,learn_rate=1,sleep_gain=0,interf=0),
  data=data[(data$transition=='42')&data$match&(data$sequence=='CoReTSeq')&(data$group!='NoReactNoInt')&(data$group!='NoReactInt')&(data$rt_pre<2),],
  group=~subject_id))

summary(interf_res<-nlme::nlme(
  model=interf_model,
  fixed=list(learn_asym+learn_start+learn_rate+sleep_gain~1,interf~group),
  random=learn_asym+learn_start+learn_rate~1,
  start=list(c(learn_asym=.2,learn_start=.5,learn_rate=1,sleep_gain=0),c(interf=0)),
  data=data[(data$transition=='23')&data$match&(data$sequence=='CoReTSeq')&(data$group!='NoReactNoInt')&(data$group!='NoReactInt')&(data$rt_pre<2),],
  group=~subject_id))

boost_func <- function(learn_asym, learn_start, learn_rate, boost_gain, task, seq_id) (
	learn_asym+(learn_start-learn_asym)*exp(-exp(learn_rate)*seq_id)+
	boost_gain*(task!='Training-TSeq-D_One')
)
boost_model<-rt_pre~boost_func(learn_asym, learn_start, learn_rate, boost_gain, task, correct_seq_idx)
summary(boost_res<-nlme::nlme(
  model=boost_model,
  fixed=learn_asym+learn_start+learn_rate+boost_gain~1,
  random=learn_asym+learn_start+learn_rate~1,
  start=list(fixed=list(learn_asym=.2,learn_start=.5,learn_rate=1,boost_gain=0)),
  data=data[(data$transition=='42')&data$match&(data$sequence=='CoReTSeq')&(data$rt_pre<2)&((data$task=='Training-TSeq-D_One')|(data$task=='TestBoost-TSeq-D_One')),],
  group=~subject_id))

boost_interf_func <- function(learn_asym, learn_start, learn_rate, boost_gain, sleep_gain, interf, task, seq_id) (
	learn_asym+(learn_start-learn_asym)*exp(-exp(learn_rate)*seq_id)+
	boost_gain*(task!='Training-TSeq-D_One')+
	sleep_gain*((task!='Training-TSeq-D_One')&(task!='TestBoost-TSeq-D_One'))+
	interf*((task!='Training-TSeq-D_One')&(task!='TestBoost-TSeq-D_One')&(task!='Reactivation-TSeq-D-Two'))
)
boost_interf_model<-rt_pre~boost_interf_func(learn_asym, learn_start, learn_rate, boost_gain, sleep_gain, interf, task, correct_seq_idx)


summary(boost_interf_res<-nlme::nlme(
  model=boost_interf_model,
  fixed=learn_asym+learn_start+learn_rate+boost_gain+sleep_gain+interf~1,
  random=learn_asym+learn_start+learn_rate~1,
  start=list(fixed=list(learn_asym=.2,learn_start=.5,learn_rate=1,boost_gain=0,sleep_gain=0,interf=0)),
  data=data[(data$transition=='42')&data$match&(data$sequence=='CoReTSeq')&(data$rt_pre<2),],
  group=~subject_id))


##### LME4 #################

model1<-lme4::nlmer(
    rt_pre~SSasymp(correct_seq_idx/max(correct_seq_idx),Asym,xmid,scale)~ (Asym+xmid+scale | subject_id),
    start=c(Asym=.2,xmid=.5,scale=.1),
    data=data[(data$transition=='14')&data$match&(data$sequence=='CoReTSeq')&(data$group!='NoReactNoInt')&(data$group!='NoReactInt'),])
summary(model1)


boost_interf_func <- deriv(
  ~learn_asym+(learn_start-learn_asym)*exp(-exp(learn_rate)*seq_id)+boost_gain*boost_mask+cons_gain*cons_mask+recons_gain*recons_mask,
               namevec=c("learn_asym","learn_start","learn_rate","boost_gain","cons_gain","recons_gain"),
               function.arg=c("learn_asym","learn_start","learn_rate","seq_id","boost_gain","cons_gain","recons_gain","boost_mask","cons_mask","recons_mask"))

cons_res_nlmer<-lme4::nlmer(
  rt_pre~boost_interf_func(
	learn_asym, learn_start, learn_rate,
	correct_seq_idx,
	boost_gain, cons_gain, recons_gain,
	(task!='Training-TSeq-D_One'),
	((task!='Training-TSeq-D_One')&(task!='TestBoost-TSeq-D_One')),
	((task!='Training-TSeq-D_One')&(task!='TestBoost-TSeq-D_One')&(task!='Reactivation-TSeq-D-Two'))
	)~ (learn_asym+learn_start+learn_rate |subject_id),
  start=list(nlpars=c(learn_asym=.2,learn_start=.5,learn_rate=1,boost_gain=0,cons_gain=0,recons_gain=0)),
  data=data[(data$transition=='14')&data$match&(data$sequence=='CoReTSeq'),],
)


boost_func <- deriv(
  ~learn_asym+(learn_start-learn_asym)*exp(-exp(learn_rate)*seq_id)+boost_gain*boost_mask,
               namevec=c("learn_asym","learn_start","learn_rate","boost_gain"),
               function.arg=c("learn_asym","learn_start","learn_rate","seq_id","boost_gain","boost_mask"))

boost_nlmer<-lme4::nlmer(
  rt_pre~boost_func(
	learn_asym, learn_start, learn_rate,
	correct_seq_idx,
	boost_gain,
	(task!='Training-TSeq-D_One')
	)~ (learn_asym+learn_start+learn_rate |subject_id),
  start=list(nlpars=c(learn_asym=.2,learn_start=.5,learn_rate=1,boost_gain=0)),
  data=data[(data$transition=='14')&data$match&(data$sequence=='CoReTSeq')&((data$task=='Training-TSeq-D_One')|(data$task=='TestBoost-TSeq-D_One')),],
)
