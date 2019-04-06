install.packages("neuralnet")
library(neuralnet)
library(caret)
setwd("F:/Data Science/MachineLearningBootcamp")
concrete <- read.csv("concrete.csv")
head(concrete)
str(concrete)
summary(concrete)
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
concrete_norm = as.data.frame(lapply(concrete, normalize))
head(concrete_norm)
summary(concrete_norm$strength)
any(is.na(concrete))
n <- nrow(concrete)
n_train <- round(0.8 * n)
set.seed(123)
train_indices <- sample(1:n, n_train)
concrete_train <- concrete_norm[train_indices, ]
concrete_test <- concrete_norm[-train_indices, ]  
dim(concrete_train)
dim(concrete_test)
set.seed(123)
concrete_model <- neuralnet(strength ~ cement + slag
                            + ash + water + superplastic + coarseagg + fineagg + age,
                            data = concrete_train)
plot(concrete_model)
model_results <- compute(concrete_model, concrete_test[1:8])
predicted_strength <- model_results$net.result
head(predicted_strength)
cor(predicted_strength, concrete_test$strength)[,1]
concrete_model_2 <- neuralnet(strength ~ cement + slag
                              + ash + water + superplastic + coarseagg + fineagg + age,
                              data = concrete_train,
                              hidden = 7)
plot(concrete_model_2)
model_results_2 <- compute(concrete_model_2, concrete_test[1:8])
predicted_strength <- model_results_2$net.result
cor(predicted_strength, concrete_test$strength)[,1]


