#load libraries (have to install them first)
library("haven") #to read SPSS files
library("foreign") #to read SPSS files
library("survival") #to do cox multivariate analysis
library("dplyr") #to make a new variable from an existing one (median TILscore to binary) 
library("survAUC")
library("jsonlite") #to write json

TIGERmainspss <- read.spss("/home/user/results.sav", to.data.frame=TRUE)
print(TIGERmainspss)

# convert variables to categorical variables (factor), R doesn't recognize this from SPSS.
TIGERmainspss$age_class <- factor(TIGERmainspss$age_class)
TIGERmainspss$morphology_group <- factor(TIGERmainspss$morphology_group)
TIGERmainspss$grade <- factor(TIGERmainspss$grade)
TIGERmainspss$molecular_group <- factor(TIGERmainspss$molecular_group)
TIGERmainspss$stage <- factor(TIGERmainspss$stage)
TIGERmainspss$surgery_breast <- factor(TIGERmainspss$surgery_breast)
TIGERmainspss$adjuvant_therapy <- factor(TIGERmainspss$adjuvant_therapy)

# =============================================
# Compute C-index and confidence interval
# =============================================
K <- 5 # Number of folds
B=500 # Number of repetitions
unorez=matrix(NA,nrow=B,ncol=K)
for(b in 1:B){
  set.seed(b)
  TIGERmainspss<-TIGERmainspss[sample(nrow(TIGERmainspss)),] #shuffle the rows
  folds <- cut(seq(from=1, to=nrow(TIGERmainspss)), breaks=K, labels=FALSE) # creates unique numbers for k equally size folds.
  TIGERmainspss$ID <- folds  # adds fold IDs.
  for(k in 1:K) { # 5-fold cross-validation # defines the model to train
    valid=subset(TIGERmainspss, TIGERmainspss$ID==k)
    train=subset(TIGERmainspss, TIGERmainspss$ID!=k)
    fitcox=coxph(Surv(time_recurrence,recurrence)~age_class+ morphology_group + grade + molecular_group + stage + surgery_breast + adjuvant_therapy + TILscore,data=train) # defines the fitting function
    predcox=predict(fitcox,newdata=valid) # does the training
    Surv.rsp=Surv(train$time_recurrence,train$recurrence) #surv on train
    Surv.rsp.new=Surv(valid$time_recurrence,valid$recurrence) #surv on valid
    unorez[b,k]=AUC.uno(Surv.rsp,Surv.rsp.new,predcox,times=5*12)$iauc #for the 5 years os prediction
  }
}
print(unorez)
print(mean(c(unorez))) #bootstrapped cv uno c-index
print(quantile(c(unorez),probs=c(0.025,0.975)))

df <- data.frame(mean(unorez), quantile(c(unorez),probs=c(0.025,0.975)))
write(toJSON(df), "/home/user/metrics.json")