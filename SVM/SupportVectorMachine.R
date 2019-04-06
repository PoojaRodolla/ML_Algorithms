setwd("E:/MachineLearningBootcamp/LinearRegression")
library(caret)
library(e1071)
db_data <- read.csv("diabetes.csv")
head(db_data)
str(db_data)
db_data$Outcome <- as.factor(db_data$Outcome)
any(is.na(db_data))
summary(db_data)

# Total number of rows in the credit data frame
n <- nrow(db_data)

# Number of rows for the training set (80% of the dataset)
n_train <- round(0.8 * n) 

# Create a vector of indices which is an 80% random sample
set.seed(123)
train_indices <- sample(1:n, n_train)

# Subset the credit data frame to training indices only
db_train <- db_data[train_indices, ]  

# Exclude the training indices to create the test set
db_test <- db_data[-train_indices, ]  

######################### using e1071 package ##############
# create model
model_e1071 <- svm(Outcome ~ . ,data=db_train,kernel='linear',gamma=0.2,cost=100)

#Predict Output
preds <- predict(model_e1071,db_test)
# Calculate the confusion matrix for the test set
confusionMatrix(data = preds,       
                reference = db_test$Outcome)  

############################ using caret package ############
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

svm_Linear <- train(Outcome ~., data = db_train, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
svm_Linear

test_pred <- predict(svm_Linear, newdata = db_test)
# Calculate the confusion matrix for the test set
confusionMatrix(data = test_pred,       
                reference = db_test$Outcome) 

########## Tuning ################
grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
svm_Linear_Grid <- train(Outcome ~., data = db_train, method = "svmLinear",
                         trControl=trctrl,
                         preProcess = c("center", "scale"),
                         tuneGrid = grid,
                         tuneLength = 10)
svm_Linear_Grid
plot(svm_Linear_Grid)

test_pred_grid <- predict(svm_Linear_Grid, newdata = db_test)
# Calculate the confusion matrix for the test set
confusionMatrix(data = test_pred_grid,       
                reference = db_test$Outcome) 

######################## Non Linear Kernel ###################
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

svm_radial <- train(Outcome ~., data = db_train, method = "svmRadial",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
svm_radial

test_pred_radial <- predict(svm_radial, newdata = db_test)
# Calculate the confusion matrix for the test set
confusionMatrix(data = test_pred_radial,       
                reference = db_test$Outcome) 





grid_radial <- expand.grid(sigma = c(0, 0.2, 0.5),
                           C = c(0, 0.05, 0.5))
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

svm_radial_grid <- train(Outcome ~., data = db_train, method = "svmRadial",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneGrid = grid_radial,
                    tuneLength = 10)
svm_radial_grid

test_pred_radial_grid <- predict(svm_radial_grid, newdata = db_test)
# Calculate the confusion matrix for the test set
confusionMatrix(data = test_pred_radial_grid,       
                reference = db_test$Outcome) 
plot(svm_radial_grid)


