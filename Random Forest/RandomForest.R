#install.packages("randomForest")
library(randomForest)
library(caret)
library(Metrics)

setwd("F:/Data Science/MachineLearningBootcamp")
db_data <- read.csv("diabetes.csv")
head(db_data)
str(db_data)
db_data$Outcome <- as.factor(db_data$Outcome)
str(db_data)
any(is.na(db_data))
summary(db_data)
n <- nrow(db_data)
n_train <- round(0.8*n)
set.seed(123)
train_indices <- sample(1:n,n_train)
db_train <- db_data[train_indices, ]
db_test <- db_data[-train_indices, ]
dim(db_train)
dim(db_test)
set.seed(123)
db_model <- randomForest(formula = Outcome ~ ., 
                         data = db_train)
print(db_model)
err <- db_model$err.rate
head(err)
oob_err <- err[nrow(err), "OOB"]
print(oob_err)
plot(db_model)
db_prediction <- predict(object = db_model,
                         newdata = db_test,
                         type = "class")
cm <- confusionMatrix(data = db_prediction,
                      reference = db_test$Outcome)
print(cm)
paste0("Test Accuracy: ", cm$overall[1])
paste0("OOB Accuracy: ", 1 - oob_err)
pred <- predict(object = db_model, 
                newdata = db_test,
                type = "prob")
class(pred)
head(pred)
auc(actual = db_test$Outcome, 
    predicted = pred[,"1"])
res <- tuneRF(x=subset(db_train, select = -Outcome),
              y=db_train$Outcome,
              ntreeTry = 500)
print(res)
mtry_opt <- res[,"mtry"][which.min(res[,"OOBError"])]
print(mtry_opt)
mtry <- seq(2, ncol(db_train), 2)
nodesize <- seq(3, 8, 2)
sampsize <- nrow(db_train) * c(0.6, 0.7, 0.8)
hyper_grid <- expand.grid(mtry = mtry, nodesize = nodesize, sampsize = sampsize)
hyper_grid
oob_err <- c()
for (i in 1:nrow(hyper_grid)) {
  model <- randomForest(formula = Outcome ~ ., 
                        data = db_train,
                        mtry = hyper_grid$mtry[i],
                        nodesize = hyper_grid$nodesize[i],
                        sampsize = hyper_grid$sampsize[i])
  oob_err[i] <- model$err.rate[nrow(model$err.rate), "OOB"]
}
oob_err
opt_i <- which.min(oob_err)
print(hyper_grid[opt_i,])
