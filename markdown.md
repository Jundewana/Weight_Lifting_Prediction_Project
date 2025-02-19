# Weight Lifting Exercise Prediction Project

#### Muhamad Arjun Dewana

#### 19/2/2025

## Step 1: Loading Libraries and Data

The first step is to load the necessary libraries and data files. These libraries are important for data manipulation, plotting, and model training.

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
# [1] 19622   160
dim(testcsv)
# [1]  20 160
```

Here, we load the training and testing datasets into R using the `read.csv()` function. We also check the number of rows and columns in both datasets using `dim()`. This gives us a sense of the size of the datasets.

## Step 2: Data Cleaning

Before we can use the data for modeling, we need to clean it. This includes removing unnecessary variables and columns with a large proportion of missing values.

```r
# Remove columns with more than 90% missing values
traincsv <- traincsv[, colMeans(is.na(traincsv)) < 0.9]

# Remove metadata columns that are irrelevant to the outcome
traincsv <- traincsv[, -c(1:7)]
```

We remove columns with a high percentage of missing values (greater than 90%) to ensure we’re working with complete data. Additionally, we remove columns such as timestamps and user names which don’t contribute to the prediction task.

## Step 3: Removing Near-Zero Variance Variables

Next, we remove variables with near-zero variance. These variables don’t provide much useful information for model training.

```r
# Identify and remove near-zero variance variables
nvz <- nearZeroVar(traincsv)
traincsv <- traincsv[, -nvz]
dim(traincsv)
# [1] 19622    53
```

The `nearZeroVar()` function identifies variables that have very little variance and are unlikely to help distinguish between different classes. We remove them to reduce the dimensionality and improve the efficiency of the model.

## Step 4: Splitting the Data into Training and Validation Sets

After cleaning the data, we split the dataset into a training set and a validation set. The validation set will be used to evaluate the model’s performance.

```r
# Split the data into training and validation sets (70% training, 30% validation)
inTrain <- createDataPartition(y = traincsv$classe, p = 0.7, list = FALSE)
train <- traincsv[inTrain, ]
valid <- traincsv[-inTrain, ]
```

Using the `createDataPartition()` function from the caret package, we split the data so that 70% of the data is used for training the model, and the remaining 30% is used as a validation set. This helps us assess how well the model generalizes to unseen data.

## Step 5: Model Training and Evaluation

Now we are ready to train multiple models and evaluate their performance. We will compare Decision Trees, Random Forests, Gradient Boosted Trees, and Support Vector Machines (SVM). We will use 3-fold cross-validation to assess each model’s performance.

```r
# Set up control for training using 3-fold cross-validation
control <- trainControl(method = "cv", number = 3, verboseIter = FALSE)
```

### Decision Tree Model

```r
# Train a Decision Tree model
mod_trees <- train(classe ~ ., data = train, method = "rpart", trControl = control, tuneLength = 5)

# Plot the decision tree
fancyRpartPlot(mod_trees$finalModel)
![Decision Tree Plot](./weight_lifting_classification_files/unnamed-chunk-7-1.png)

pred_trees <- predict(mod_trees, valid)
cmtrees <- confusionMatrix(pred_trees, factor(valid$classe))
cmtrees
```

A Decision Tree model is trained using the `rpart` method. We visualize the tree using `fancyRpartPlot()`. This model is then evaluated using the validation set.

### Random Forest Model

```r
# Train a Random Forest model
mod_rf <- train(classe ~ ., data = train, method = "rf", trControl = control, tuneLength = 5)

# Make predictions and evaluate the Random Forest model
pred_rf <- predict(mod_rf, valid)
cmrf <- confusionMatrix(pred_rf, factor(valid$classe))
cmrf
```

The Random Forest model is trained using the `rf` method. After training, we make predictions on the validation set and calculate the accuracy using the `confusionMatrix()` function from caret.

### Gradient Boosted Trees Model

```r
# Train a Gradient Boosted Trees model
mod_gbm <- train(classe ~ ., data = train, method = "gbm", trControl = control, tuneLength = 5, verbose = FALSE)

# Make predictions and evaluate the Gradient Boosted Trees model
pred_gbm <- predict(mod_gbm, valid)
cmgbm <- confusionMatrix(pred_gbm, factor(valid$classe))
cmgbm
```

A Gradient Boosted Trees model is trained using the `gbm` method. We make predictions and evaluate the model’s accuracy in the same way as the previous models.

### Support Vector Machine Model

```r
# Train a Support Vector Machine model
mod_svm <- train(classe ~ ., data = train, method = "svmLinear", trControl = control, tuneLength = 5, verbose = FALSE)

# Make predictions and evaluate the Support Vector Machine model
pred_svm <- predict(mod_svm, valid)
cmsvm <- confusionMatrix(pred_svm, factor(valid$classe))
cmsvm
```

We train a Support Vector Machine (SVM) model using the `svmLinear` method. After training, we evaluate the model’s accuracy using the validation set.

## Step 6: Evaluating Model Accuracy and Out of Sample Error

Now that we have trained and evaluated the models, let’s compare their accuracy and out-of-sample error (the error on the validation set).

```r
# Collect the accuracy of each model
models <- c("Tree", "RF", "GBM", "SVM")
accuracy <- round(c(cmtrees$overall[1], cmrf$overall[1], cmgbm$overall[1], cmsvm$overall[1]), 3) # accuracy
oos_error <- 1 - accuracy # out-of-sample error

# Display the results
data.frame(accuracy = accuracy, oos_error = oos_error, row.names = models)
```

We calculate and display the accuracy and out-of-sample error for each model. This helps us determine which model performs best on the validation set.

## Step 7: Making Predictions on the Test Set

Finally, we use the Random Forest model to make predictions for the 20 test cases provided in the test set.

```r
# Predict the classe for 20 test cases
pred <- predict(mod_rf, testcsv)
print(pred)
# [1] B A B A A E D B A A B C B A E E A B B B
# Levels: A B C D E
```

Here, we use the Random Forest model to predict the “classe” variable for the 20 test cases in the `testcsv` dataset. These predictions will be submitted to the course project prediction quiz.

```html
<script>
    // Add bootstrap table styles to pandoc tables
    function bootstrapStylePandocTables() {
        $('tr.odd').parent('tbody').parent('table').addClass('table table-condensed');
    }
    $(document).ready(function () {
        bootstrapStylePandocTables();
    });
    $(document).ready(function () {
        window.buildTabsets("TOC");
    });
    $(document).ready(function () {
        $('.tabset-dropdown > .nav-tabs > li').click(function () {
            $(this).parent().toggleClass('nav-tabs-open');
        });
    });
    (function () {
        var script = document.createElement("script");
        script.type = "text/javascript";
        script.src = "https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML";
        document.getElementsByTagName("head")[0].appendChild(script);
    })();
</script>
```
