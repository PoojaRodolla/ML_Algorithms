library(Amelia)
library(ggplot2)
library(tidyverse)
library(caTools)

df.train <- read.csv("F:/Data Science/csvUdemy/titanic_train.csv")
head(df.train)

#EDA
missmap(df.train, main = "Titanic training data-Missing Maps",
        col=c("yellow","black"),legend = FALSE)
ggplot(df.train,aes(Survived)) + geom_bar()
ggplot(df.train,aes(Pclass)) + geom_bar(aes(fill=factor(Pclass)),alpha=0.5)
ggplot(df.train,aes(Sex)) + geom_bar(aes(fill=factor(Sex)),alpha=0.5)
ggplot(df.train,aes(Age)) + geom_histogram(fill='blue',bins = 20,alpha=0.5)
ggplot(df.train,aes(SibSp)) +geom_bar(fill='red',alpha=0.5)
ggplot(df.train,aes(Fare)) + geom_histogram(fill='green',color='black',alpha=0.5)

#Data cleaning
pl <- ggplot(df.train,aes(Pclass,Age)) + geom_boxplot(aes(group=Pclass,
                           fill=factor(Pclass),alpha=0.4))
pl + scale_y_continuous(breaks = seq(min(0),max(80),by=2))

impute_age <- function(age,class){
  out <- age
  for (i in 1:length(age)) {
    
    if (is.na(age[i])){
    
        if (class[i]==1){
        out[i] <- 37
      
        }else if (class[i]==2){
        out[i] <- 29
      
        }else {
        out[i] <- 24
      }
    
      }else {
      out[i] <- age[i]
    }
    
  }
  return(out)
}
fixed.age <- impute_age(df.train$Age,df.train$Pclass)
df.train$Age <- fixed.age
missmap(df.train, main = "Titanic training data-Missing Maps",
        col=c("yellow","black"),legend = FALSE)

#model
str(df.train)
head(df.train,3)
df.train <- select(df.train, -PassengerId, -Name, -Ticket,-Cabin)
head(df.train,3)
str(df.train)

df.train$Survived <- factor(df.train$Survived)
df.train$Pclass <- factor(df.train$Pclass)
df.train$Parch <- factor(df.train$Parch)
df.train$SibSp <- factor(df.train$SibSp)
str(df.train)

log.model<- glm(Survived~., family = binomial(link ='logit'),data=df.train)
summary(log.model)

set.seed(101)
split=sample.split(df.train$Survived, SplitRatio = 0.7)
final.train =subset(df.train, split=TRUE)
final.test =subset(df.train,split=FALSE)
final.log.model<- glm(Survived~., family = binomial(link ='logit'),data=df.train)
summary(final.log.model)

fitted.probabilities <- predict(final.log.model, newdata = final.test,
                                type = 'response')
fitted.result <- ifelse(fitted.probabilities >0.5,1,0)
misClasificError <- mean(fitted.result!= final.test$Survived)
print(paste('Accuracy',1-misClasificError))
table(final.test$Survived, fitted.probabilities>0.5)
