Step 1: Loading Libraries and Data
----------------------------------

The first step is to load the necessary libraries and data files. These
libraries are important for data manipulation, plotting, and model
training.

``` r
# Load essential libraries
library(lattice)
library(ggplot2)
library(caret)
library(kernlab)
library(rattle)
library(corrplot)
# Set the random seed for reproducibility
set.seed(1234)
```

``` r
# Load the training and testing datasets
traincsv <- read.csv("./project/pml-training.csv")
testcsv <- read.csv("./project/pml-testing.csv")

# Check the dimensions of the datasets
dim(traincsv)
```

    ## [1] 19622   160

``` r
dim(testcsv)
```

    ## [1]  20 160

Here, we load the training and testing datasets into R using the
read.csv() function. We also check the number of rows and columns in
both datasets using dim(). This gives us a sense of the size of the
datasets

Step 2. Data Cleaning
---------------------

Before we can use the data for modeling, we need to clean it. This
includes removing unnecessary variables and columns with a large
proportion of missing values.

``` r
# Remove columns with more than 90% missing values
traincsv <- traincsv[, colMeans(is.na(traincsv)) < 0.9]

# Remove metadata columns that are irrelevant to the outcome
traincsv <- traincsv[, -c(1:7)]
```

We remove columns with a high percentage of missing values (greater than
90%) to ensure we’re working with complete data. Additionally, we remove
columns such as timestamps and user names which don’t contribute to the
prediction task.

\#\#Step 3: Removing Near-Zero Variance Variables Next, we remove
variables with near-zero variance. These variables don’t provide much
useful information for model training.

``` r
# Identify and remove near-zero variance variables
nvz <- nearZeroVar(traincsv)
traincsv <- traincsv[, -nvz]
dim(traincsv)
```

    ## [1] 19622    53

The nearZeroVar() function identifies variables that have very little
variance and are unlikely to help distinguish between different classes.
We remove them to reduce the dimensionality and improve the efficiency
of the model.

\#\#Step 4: Splitting the Data into Training and Validation Sets After
cleaning the data, we split the dataset into a training set and a
validation set. The validation set will be used to evaluate the model’s
performance.

``` r
# Split the data into training and validation sets (70% training, 30% validation)
inTrain <- createDataPartition(y = traincsv$classe, p = 0.7, list = FALSE)
train <- traincsv[inTrain, ]
valid <- traincsv[-inTrain, ]
```

Using the createDataPartition() function from the caret package, we
split the data so that 70% of the data is used for training the model,
and the remaining 30% is used as a validation set. This helps us assess
how well the model generalizes to unseen data.

\#\#Step 5: Model Training and Evaluation Now we are ready to train
multiple models and evaluate their performance. We will compare Decision
Trees, Random Forests, Gradient Boosted Trees, and Support Vector
Machines (SVM). We will use 3-fold cross-validation to assess each
model’s performance.

``` r
# Set up control for training using 3-fold cross-validation
control <- trainControl(method = "cv", number = 3, verboseIter = FALSE)
```

\#\#\#Decision Tree Model

``` r
# Train a Decision Tree model
mod_trees <- train(classe ~ ., data = train, method = "rpart", trControl = control, tuneLength = 5)

# Plot the decision tree
fancyRpartPlot(mod_trees$finalModel)
```

![](weight_lifting_classification_files/figure-markdown_github/unnamed-chunk-7-1.png)
\# Make predictions using the Decision Tree model

``` r
pred_trees <- predict(mod_trees, valid)
cmtrees <- confusionMatrix(pred_trees, factor(valid$classe))
cmtrees
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1519  473  484  451  156
    ##          B   28  355   45   10  130
    ##          C   83  117  423  131  131
    ##          D   40  194   74  372  176
    ##          E    4    0    0    0  489
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.5366          
    ##                  95% CI : (0.5238, 0.5494)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3957          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9074  0.31168  0.41228  0.38589  0.45194
    ## Specificity            0.6286  0.95512  0.90492  0.90165  0.99917
    ## Pos Pred Value         0.4927  0.62500  0.47797  0.43458  0.99189
    ## Neg Pred Value         0.9447  0.85255  0.87940  0.88228  0.89002
    ## Prevalence             0.2845  0.19354  0.17434  0.16381  0.18386
    ## Detection Rate         0.2581  0.06032  0.07188  0.06321  0.08309
    ## Detection Prevalence   0.5239  0.09652  0.15038  0.14545  0.08377
    ## Balanced Accuracy      0.7680  0.63340  0.65860  0.64377  0.72555

A Decision Tree model is trained using the rpart method. We visualize
the tree using fancyRpartPlot(). This model is then evaluated using the
validation set.

\#\#\#Random Forest Model

``` r
# Train a Random Forest model
mod_rf <- train(classe ~ ., data = train, method = "rf", trControl = control, tuneLength = 5)

# Make predictions and evaluate the Random Forest model
pred_rf <- predict(mod_rf, valid)
cmrf <- confusionMatrix(pred_rf, factor(valid$classe))
cmrf
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673    4    0    0    0
    ##          B    1 1132    8    0    0
    ##          C    0    3 1016    5    1
    ##          D    0    0    2  958    0
    ##          E    0    0    0    1 1081
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9958          
    ##                  95% CI : (0.9937, 0.9972)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9946          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9994   0.9939   0.9903   0.9938   0.9991
    ## Specificity            0.9991   0.9981   0.9981   0.9996   0.9998
    ## Pos Pred Value         0.9976   0.9921   0.9912   0.9979   0.9991
    ## Neg Pred Value         0.9998   0.9985   0.9979   0.9988   0.9998
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2843   0.1924   0.1726   0.1628   0.1837
    ## Detection Prevalence   0.2850   0.1939   0.1742   0.1631   0.1839
    ## Balanced Accuracy      0.9992   0.9960   0.9942   0.9967   0.9994

The Random Forest model is trained using the rf method. After training,
we make predictions on the validation set and calculate the accuracy
using the confusionMatrix() function from caret.

\#\#\#Gradient Boosted Trees Model

``` r
# Train a Gradient Boosted Trees model
mod_gbm <- train(classe ~ ., data = train, method = "gbm", trControl = control, tuneLength = 5, verbose = FALSE)

# Make predictions and evaluate the Gradient Boosted Trees model
pred_gbm <- predict(mod_gbm, valid)
cmgbm <- confusionMatrix(pred_gbm, factor(valid$classe))
cmgbm
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1671    6    0    0    0
    ##          B    1 1126   10    0    0
    ##          C    2    7 1013    9    2
    ##          D    0    0    3  953    2
    ##          E    0    0    0    2 1078
    ## 
    ## Overall Statistics
    ##                                         
    ##                Accuracy : 0.9925        
    ##                  95% CI : (0.99, 0.9946)
    ##     No Information Rate : 0.2845        
    ##     P-Value [Acc > NIR] : < 2.2e-16     
    ##                                         
    ##                   Kappa : 0.9905        
    ##                                         
    ##  Mcnemar's Test P-Value : NA            
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9982   0.9886   0.9873   0.9886   0.9963
    ## Specificity            0.9986   0.9977   0.9959   0.9990   0.9996
    ## Pos Pred Value         0.9964   0.9903   0.9806   0.9948   0.9981
    ## Neg Pred Value         0.9993   0.9973   0.9973   0.9978   0.9992
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2839   0.1913   0.1721   0.1619   0.1832
    ## Detection Prevalence   0.2850   0.1932   0.1755   0.1628   0.1835
    ## Balanced Accuracy      0.9984   0.9931   0.9916   0.9938   0.9979

A Gradient Boosted Trees model is trained using the gbm method. We make
predictions and evaluate the model’s accuracy in the same way as the
previous models.

\#\#\#Support Vector Machine Model

``` r
# Train a Support Vector Machine model
mod_svm <- train(classe ~ ., data = train, method = "svmLinear", trControl = control, tuneLength = 5, verbose = FALSE)

# Make predictions and evaluate the Support Vector Machine model
pred_svm <- predict(mod_svm, valid)
cmsvm <- confusionMatrix(pred_svm, factor(valid$classe))
cmsvm
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1537  154   79   69   50
    ##          B   29  806   90   46  152
    ##          C   40   81  797  114   69
    ##          D   61   22   32  697   50
    ##          E    7   76   28   38  761
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7813          
    ##                  95% CI : (0.7705, 0.7918)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.722           
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9182   0.7076   0.7768   0.7230   0.7033
    ## Specificity            0.9164   0.9332   0.9374   0.9665   0.9690
    ## Pos Pred Value         0.8137   0.7177   0.7239   0.8086   0.8363
    ## Neg Pred Value         0.9657   0.9301   0.9521   0.9468   0.9355
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2612   0.1370   0.1354   0.1184   0.1293
    ## Detection Prevalence   0.3210   0.1908   0.1871   0.1465   0.1546
    ## Balanced Accuracy      0.9173   0.8204   0.8571   0.8447   0.8362

We train a Support Vector Machine (SVM) model using the svmLinear
method. After training, we evaluate the model’s accuracy using the
validation set.

\#\#Step 6: Evaluating Model Accuracy and Out of Sample Error Now that
we have trained and evaluated the models, let’s compare their accuracy
and out-of-sample error (the error on the validation set).

``` r
# Collect the accuracy of each model
models <- c("Tree", "RF", "GBM", "SVM")
accuracy <- round(c(cmtrees$overall[1], cmrf$overall[1], cmgbm$overall[1], cmsvm$overall[1]), 3) # accuracy
oos_error <- 1 - accuracy # out-of-sample error

# Display the results
data.frame(accuracy = accuracy, oos_error = oos_error, row.names = models)
```

    ##      accuracy oos_error
    ## Tree    0.537     0.463
    ## RF      0.996     0.004
    ## GBM     0.993     0.007
    ## SVM     0.781     0.219

We calculate and display the accuracy and out-of-sample error for each
model. This helps us determine which model performs best on the
validation set.

\#\#Step 7: Making Predictions on the Test Set Finally, we use the
Random Forest model to make predictions for the 20 test cases provided
in the test set.

``` r
# Predict the classe for 20 test cases
pred <- predict(mod_rf, testcsv)
print(pred)
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

Here, we use the Random Forest model to predict the “classe” variable
for the 20 test cases in the testcsv dataset. These predictions will be
submitted to the course project prediction quiz.
