# -------------------------------
# Offline Optimized Random Forest
# -------------------------------

# 1. Packages
library(caret)
library(randomForest)
library(doParallel)

# 2. Load data
train_raw <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
test_raw  <- read.csv("pml-testing.csv",  na.strings=c("NA","#DIV/0!",""))

# 3. Clean columns: remove mostly-NA and identifiers
na_frac <- sapply(train_raw, function(x) mean(is.na(x)))
keep_cols <- names(na_frac[na_frac <= 0.6])
drop_names <- c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2",
                "cvtd_timestamp","new_window","num_window")
keep_cols <- setdiff(keep_cols, drop_names)

train <- train_raw[, keep_cols]
test  <- test_raw[, keep_cols[keep_cols != "classe"]]  # test has no 'classe'

# 4. Remove near-zero variance predictors
nzv <- nearZeroVar(train)
if(length(nzv) > 0) train <- train[, -nzv]

# 5. Optional: remove highly correlated numeric predictors
numVars <- sapply(train, is.numeric)
corrMatrix <- cor(train[, numVars], use="pairwise.complete.obs")
highCorr <- findCorrelation(corrMatrix, cutoff = 0.9)
train <- train[, !(names(train) %in% names(train)[numVars][highCorr])]

# 6. Train/Validation split
set.seed(3456)
inTrain <- createDataPartition(train$classe, p = 0.7, list = FALSE)
trainPart <- train[inTrain, ]
validPart <- train[-inTrain, ]

# 7. Parallel backend
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# 8. Train Random Forest with cross-validation
set.seed(100)
rfFit <- train(classe ~ ., data = trainPart,
               method = "rf",
               trControl = trainControl(method="cv", number=5),
               ntree = 500,
               importance = TRUE)

stopCluster(cl)  # stop parallel cluster

# 9. Validation accuracy
rfPred <- predict(rfFit, validPart)
print(confusionMatrix(rfPred, validPart$classe))

# 10. Predict 20-case test set
finalPred <- predict(rfFit, test)

# 11. Write predictions in required format (20 separate files)
pml_write_files <- function(x){
  n <- length(x)
  for(i in 1:n){
    fname <- paste0("problem_id_", i, ".txt")
    write.table(x[i], file = fname, quote = FALSE, row.names = FALSE, col.names = FALSE)
  }
}
pml_write_files(finalPred)

# Optional: write all predictions to one CSV
write.csv(data.frame(Id = 1:length(finalPred), Prediction = finalPred),
          "final_predictions.csv", row.names = FALSE)

# -------------------------------
# Done
# -------------------------------

