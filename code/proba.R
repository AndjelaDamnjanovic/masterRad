# Load libraries
library(tidyverse)
library(Biostrings)  # For reading FASTA
library(lightgbm)
library(DMwR)        # For SMOTE
library(caret)       # For confusion matrix and encoding

# 1️⃣ Read CSV files with repeats
folder <- "proba3/IN/"
files <- list.files(folder, full.names = TRUE)
files <- files[!grepl("^\\.", basename(files))]

labels <- c()
repeats <- c()

for (file in files) {
  lines <- readLines(file)
  for (line in lines) {
    parts <- strsplit(line, ",")[[1]]
    if (length(parts) >= 2) {
      labels <- c(labels, parts[1])
      repeats <- c(repeats, parts[2])
    }
  }
}

# 2️⃣ Read training FASTA
folder_train <- "trainSetNucl/"
files_train <- list.files(folder_train, full.names = TRUE)
files_train <- files_train[!grepl("^\\.", basename(files_train))]

sequencesTrain <- c()
labelTrain <- c()

for (file in files_train) {
  records <- readDNAStringSet(file)
  sequencesTrain <- c(sequencesTrain, as.character(records))
  label <- gsub("\\.fasta$", "", basename(file))
  labelTrain <- c(labelTrain, rep(label, length(records)))
}

# 3️⃣ Read testing FASTA
folder_test <- "testSetNucl/"
files_test <- list.files(folder_test, full.names = TRUE)
files_test <- files_test[!grepl("^\\.", basename(files_test))]

sequencesTest <- c()
labelTest <- c()

for (file in files_test) {
  records <- readDNAStringSet(file)
  sequencesTest <- c(sequencesTest, as.character(records))
  label <- gsub("\\.fasta$", "", basename(file))
  labelTest <- c(labelTest, rep(label, length(records)))
}

# 4️⃣ Extract known repeats
known_repeats <- unique(repeats)
cat("Number of unique repeats: ", length(known_repeats), "\n")

# 5️⃣ Extract repeat counts
extract_repeat_counts <- function(seq, repeat_list) {
  sapply(repeat_list, function(r) str_count(seq, fixed(r)))
}

X_train <- t(sapply(sequencesTrain, extract_repeat_counts, repeat_list = known_repeats))
X_test <- t(sapply(sequencesTest, extract_repeat_counts, repeat_list = known_repeats))

y_train <- as.factor(labelTrain)
y_test <- as.factor(labelTest)

# 6️⃣ Apply SMOTE
data_train <- as.data.frame(X_train)
data_train$label <- y_train

# SMOTE needs factor target variable
data_train$label <- as.factor(data_train$label)

# Apply SMOTE
data_train_smote <- SMOTE(label ~ ., data = data_train, perc.over = 200, perc.under = 100, k = 2)

X_train_res <- data_train_smote %>% select(-label) %>% as.matrix()
y_train_res <- data_train_smote$label

# 7️⃣ Train LightGBM
# LightGBM needs numeric target labels starting from 0
y_train_res_num <- as.numeric(y_train_res) - 1
y_test_num <- as.numeric(y_test) - 1

lgb_train <- lgb.Dataset(X_train_res, label = y_train_res_num)

params <- list(
  objective = "multiclass",
  num_class = length(unique(y_train_res)),
  metric = "multi_logloss",
  learning_rate = 0.1,
  is_unbalance = TRUE,
  verbose = -1
)

model <- lgb.train(
  params,
  lgb_train,
  nrounds = 100
)

# 8️⃣ Predict
y_pred_probs <- predict(model, X_test)
y_pred_matrix <- matrix(y_pred_probs, ncol = length(unique(y_train_res)), byrow = TRUE)
y_pred_num <- max.col(y_pred_matrix) - 1

# Map numeric prediction back to factor levels
levels_map <- levels(y_train_res)
y_pred <- factor(levels_map[y_pred_num + 1], levels = levels(y_test))

# 9️⃣ Report
cat("Classification report:\n")
print(confusionMatrix(y_pred, y_test))

accuracy <- mean(y_pred == y_test)
cat("Accuracy score: ", accuracy, "\n")
