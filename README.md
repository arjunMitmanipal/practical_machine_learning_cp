# Practical Machine Learning Project

## Project Overview

This project is part of the Coursera *Practical Machine Learning* course. The goal is to predict the manner in which participants performed exercises using data collected from accelerometers on the belt, forearm, arm, and dumbell. The outcome variable to predict is `classe`, which represents the type of exercise performance.

The dataset consists of:

- **Training data:** `pml-training.csv` (contains 19622 observations and multiple sensor measurements)
- **Testing data:** `pml-testing.csv` (contains 20 observations, without `classe`)

The project involves cleaning the data, training machine learning models, validating their performance, and generating predictions for the 20-case test set.

---

## Data Cleaning and Preprocessing

1. **Remove columns with mostly missing values (>60% NA).**  
2. **Remove identifier and timestamp columns**: `X`, `user_name`, `raw_timestamp_part_1`, `raw_timestamp_part_2`, `cvtd_timestamp`, `new_window`, `num_window`.  
3. **Remove near-zero variance predictors** using `caret::nearZeroVar()`.  
4. **Remove highly correlated numeric predictors** with correlation > 0.9.  
5. Split the training data into **training (70%)** and **validation (30%)** sets.  
6. Ensure the outcome variable `classe` is a factor for training and validation.

---

## Model Training

- **Random Forest (`rf`)** was used for classification.  
- Parallel processing was enabled using `doParallel` for faster training.  
- Cross-validation (5-fold CV) was used to estimate out-of-sample performance.  
- The model was trained on the cleaned training data.

Optional: The pipeline can be extended with **GBM or stacking** for improved accuracy.

---

## Validation

- Predictions were made on the validation set.  
- Performance was measured using **accuracy and confusion matrix**.  

---

## Test Set Predictions

- Predictions were made on the 20-case test set (`pml-testing.csv`) using the trained Random Forest model.  
- Coursera requires **20 separate `.txt` files**, each containing a single prediction.  
- An optional combined CSV file with all predictions is also generated (`final_predictions.csv`).

---

## How to Run

1. Ensure the following R packages are installed:
   ```r
   install.packages(c("caret", "randomForest", "doParallel", "dplyr"))
