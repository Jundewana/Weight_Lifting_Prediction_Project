# Load essential libraries
library(lattice)
library(ggplot2)
library(caret)
library(kernlab)
library(rattle)
library(corrplot)

# Set the random seed for reproducibility
set.seed(1234)

# Load the training and testing datasets
traincsv <- read.csv("./project/pml-training.csv")
testcsv <- read.csv("./project/pml-testing.csv")

# Check the dimensions of the datasets
dim(traincsv)
dim(testcsv)

# Remove columns with more than 90% missing values
traincsv <- traincsv[, colMeans(is.na(traincsv)) < 0.9]

# Remove metadata columns that are irrelevant to the outcome
traincsv <- traincsv[, -c(1:7)]

# Identify and remove near-zero variance variables
nvz <- nearZeroVar(traincsv)
traincsv <- traincsv[, -nvz]
dim(traincsv)

# Split the data into training and validation sets (70% training, 30% validation)
inTrain <- createDataPartition(y = traincsv$classe, p = 0.7, list = FALSE)
train <- traincsv[inTrain, ]
valid <- traincsv[-inTrain, ]

# Set up control for training using 3-fold cross-validation
control <- trainControl(method = "cv", number = 3, verboseIter = FALSE)

# Train a Decision Tree model
mod_trees <- train(classe ~ ., data = train, method = "rpart", trControl = control, tuneLength = 5)

# Plot the decision tree
fancyRpartPlot(mod_trees$finalModel)

# Make predictions using the Decision Tree model
pred_trees <- predict(mod_trees, valid)
cmtrees <- confusionMatrix(pred_trees, factor(valid$classe))
cmtrees

# Train a Random Forest model
mod_rf <- train(classe ~ ., data = train, method = "rf", trControl = control, tuneLength = 5)

# Make predictions and evaluate the Random Forest model
pred_rf <- predict(mod_rf, valid)
cmrf <- confusionMatrix(pred_rf, factor(valid$classe))
cmrf

# Train a Gradient Boosted Trees model
mod_gbm <- train(classe ~ ., data = train, method = "gbm", trControl = control, tuneLength = 5, verbose = FALSE)

# Make predictions and evaluate the Gradient Boosted Trees model
pred_gbm <- predict(mod_gbm, valid)
cmgbm <- confusionMatrix(pred_gbm, factor(valid$classe))
cmgbm

# Train a Support Vector Machine model
mod_svm <- train(classe ~ ., data = train, method = "svmLinear", trControl = control, tuneLength = 5, verbose = FALSE)

# Make predictions and evaluate the Support Vector Machine model
pred_svm <- predict(mod_svm, valid)
cmsvm <- confusionMatrix(pred_svm, factor(valid$classe))
cmsvm

# Collect the accuracy of each model
models <- c("Tree", "RF", "GBM", "SVM")
accuracy <- round(c(cmtrees$overall[1], cmrf$overall[1], cmgbm$overall[1], cmsvm$overall[1]), 3) # accuracy
oos_error <- 1 - accuracy # out-of-sample error

# Display the results
data.frame(accuracy = accuracy, oos_error = oos_error, row.names = models)

# Predict the classe for 20 test cases
pred <- predict(mod_rf, testcsv)
print(pred)

# Plot the correlation matrix for the training features
corrPlot <- cor(train[, -length(names(train))])
corrplot(corrPlot, method = "color")
