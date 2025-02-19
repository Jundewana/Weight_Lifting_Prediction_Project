# Weight Lifting Exercise Prediction Project

**Author:** Muhamad Arjun Dewana  
**Date:** 19/2/2025

## Step 1: Loading Libraries and Data

The first step is to load the necessary libraries and data files. These libraries are important for data manipulation, plotting, and model training.

### Code Input
```r
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
```

### Code Output
```
## [1] 19622   160
## [1]  20 160
```

Here, we load the training and testing datasets into R using the `read.csv()` function. We also check the number of rows and columns in both datasets using `dim()`. This gives us a sense of the size of the datasets.

## Step 2: Data Cleaning

Before we can use the data for modeling, we need to clean it. This includes removing unnecessary variables and columns with a large proportion of missing values.

### Code Input
```r
# Remove columns with more than 90% missing values
traincsv <- traincsv[, colMeans(is.na(traincsv)) < 0.9]

# Remove metadata columns that are irrelevant to the outcome
traincsv <- traincsv[, -c(1:7)]
```

We remove columns with a high percentage of missing values (greater than 90%) to ensure we're working with complete data. Additionally, we remove columns such as timestamps and user names which don't contribute to the prediction task.

## Step 3: Removing Near-Zero Variance Variables

Next, we remove variables with near-zero variance. These variables don't provide much useful information for model training.

### Code Input
```r
# Identify and remove near-zero variance variables
nvz <- nearZeroVar(traincsv)
traincsv <- traincsv[, -nvz]
dim(traincsv)
```

### Code Output
```
## [1] 19622    53
```

The `nearZeroVar()` function identifies variables that have very little variance and are unlikely to help distinguish between different classes. We remove them to reduce the dimensionality and improve the efficiency of the model.

## Step 4: Splitting the Data into Training and Validation Sets

After cleaning the data, we split the dataset into a training set and a validation set. The validation set will be used to evaluate the model's performance.

### Code Input
```r
# Split the data into training and validation sets (70% training, 30% validation)
inTrain <- createDataPartition(y = traincsv$classe, p = 0.7, list = FALSE)
train <- traincsv[inTrain, ]
valid <- traincsv[-inTrain, ]
```

Using the `createDataPartition()` function from the caret package, we split the data so that 70% of the data is used for training the model, and the remaining 30% is used as a validation set. This helps us assess how well the model generalizes to unseen data.

## Step 5: Model Training and Evaluation

Now we are ready to train multiple models and evaluate their performance. We will compare Decision Trees, Random Forests, Gradient Boosted Trees, and Support Vector Machines (SVM). We will use 3-fold cross-validation to assess each model's performance.

### Code Input
```r
# Set up control for training using 3-fold cross-validation
control <- trainControl(method = "cv", number = 3, verboseIter = FALSE)
```

(The rest of the model training and evaluation code follows a similar format, with code inputs and outputs separated)

## Step 6: Evaluating Model Accuracy and Out of Sample Error

### Code Input
```r
# Collect the accuracy of each model
models <- c("Tree", "RF", "GBM", "SVM")
accuracy <- round(c(cmtrees$overall[1], cmrf$overall[1], cmgbm$overall[1], cmsvm$overall[1]), 3) # accuracy
oos_error <- 1 - accuracy # out-of-sample error

# Display the results
data.frame(accuracy = accuracy, oos_error = oos_error, row.names = models)
```

### Code Output
```
##      accuracy oos_error
## Tree    0.537     0.463
## RF      0.996     0.004
## GBM     0.993     0.007
## SVM     0.781     0.219
```

## Step 7: Making Predictions on the Test Set

### Code Input
```r
# Predict the classe for 20 test cases
pred <- predict(mod_rf, testcsv)
print(pred)
```

### Code Output
```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

Here, we use the Random Forest model to predict the "classe" variable for the 20 test cases in the testcsv dataset. These predictions will be submitted to the course project prediction quiz.
