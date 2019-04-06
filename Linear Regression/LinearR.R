library(tidyverse)
library(broom)
library(caret)
library(psych)
library(corrplot)
library(caTools)
cars <- read.csv("F:/Data Science/Data/cars2018.csv")
glimpse(cars)
summary(cars)
head(cars)
any(is.na(cars))
# Grab only numeric columns
num.cols <- sapply(cars, is.numeric)
# Filter to numeric columns for correlation
cor.data <- cor(cars[,num.cols])
cor.data
corrplot(cor.data, method = 'color')

pairs(cars[,num.cols])
pairs.panels(cars[,num.cols])

# Let us try to visualize the feature which we are about to predict.
ggplot(cars, aes(x = MPG)) + 
  geom_histogram(bins = 25) +
  labs(y = "Number of Cars", x = "Fuel Efficiency (mpg)")

cars_vars <- cars %>% select(-Model, -Model.Index)

set.seed(1234)
sample <- createDataPartition(cars_vars$Transmission, p = 0.8, list = FALSE)
training <- cars_vars[sample, ]
testing <- cars_vars[-sample, ]
model <- train(MPG ~ ., method = "lm", data = training,
               trControl = trainControl(method = "none"))
summary(model)
res <- as.data.frame(residuals(model))
head(res)

ggplot(res,aes(res)) + geom_histogram(fill = 'blue', alpha = 0.5)

MPGPredictions <- predict(model, testing)
result <- as.data.frame(cbind(MPGPredictions,testing$MPG))
colnames(result) <- c('pred','real')
head(result)

### Mean Squared Error

mse <- mean((result$real - result$pred)^2)
print(mse)

### Root mean squared error

mse^0.5

##### R Squared

SSE = sum((result$pred - result$real)^2)
SST = sum( (mean(cars_vars$MPG) - result$real)^2)

R2 = 1 - SSE/SST
R2
