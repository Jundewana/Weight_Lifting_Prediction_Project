# Weight Lifting Exercise Prediction Project

## Project Overview

This project is part of a machine learning course where we are tasked with predicting the manner in which individuals perform weight lifting exercises. The dataset includes sensor data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants.

The main objective is to predict the **"classe"** variable, which indicates how the exercise was performed. We use various machine learning techniques to build the predictive model and evaluate its accuracy.

## Dataset

The dataset is from the **Human Activity Recognition (HAR) project** and includes data from different exercise conditions. There are two main datasets:
- **Training Data:** This dataset is used to train the model.
  - File: `pml-training.csv`
- **Testing Data:** This dataset is used to test the model and predict the performance on unseen data.
  - File: `pml-testing.csv`

## Project Steps

### 1. Data Preparation
- **Loading and Inspecting Data:** Load the training and testing data, check the structure and dimensions.
- **Data Cleaning:** Remove irrelevant and missing values from the dataset.
- **Feature Selection:** Separate the target variable "classe" and features.

### 2. Model Building
- **Random Forest:** A Random Forest model is trained using the features from the dataset.
- **Boosting:** Gradient Boosting Machines (GBM) are used to train another predictive model.

### 3. Model Evaluation
- **Cross-Validation:** Used to prevent overfitting and check the model's ability to generalize.
- **Model Accuracy:** Evaluation of model accuracy on both training and testing datasets.

### 4. Making Predictions
- **Test Predictions:** The model is used to predict the classes for 20 test cases in the testing dataset.

## Files

- **weight_lifting_classification.Rmd:** This file contains the R Markdown code used for the analysis. It includes the steps for data preparation, model building, and evaluation.
- **output.md:** This file contains the output from the R Markdown analysis, providing detailed steps and results of the data analysis and model training.
- **index.html:** This file is the HTML output generated from the R Markdown file. It provides a web-based view of the analysis and results.

## Deployment

The web-based view of the analysis can be accessed via the following link: [Weight Lifting Prediction Project Deployment](https://jundewana.github.io/Weight_Lifting_Prediction_Project/)

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/Jundewana/Weight_Lifting_Prediction_Project.git
