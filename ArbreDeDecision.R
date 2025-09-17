
train <- read.csv(choose.files())
test  <- read.csv(choose.files())

str(train)
summary(train)



# Gestion des NA
colSums(is.na(train))


set.seed(42)
idx <- sample(seq_len(nrow(train)), size = 0.8 * nrow(train))
train_set <- train[idx, ]
valid_set <- train[-idx, ]

#l'arbre de decision 
install.packages("rpart") 
install.packages("rpart.plot", repos = "https://cloud.r-project.org")
install.packages("recipes", repos = "https://cloud.r-project.org")
install.packages("caret")
library(rpart.plot)
library(caret)
library(recipes)
tree_fit <- rpart(DIFF ~ ., data=train_set, method="class", cp=0.01)
rpart.plot::rpart.plot(tree_fit)
pred_tree <- predict(tree_fit, valid_set, type="class")
tab <- table(Prediction = pred_tree, Réel = valid_set$DIFF)
print(tab)
# Taux de bonne classification (accuracy)
accuracy <- sum(diag(tab)) / sum(tab)
cat("Accuracy :", round(accuracy, 4), "\n")




library(rpart)


train$DIFF <- as.factor(train$DIFF)

# Entraînement sur tout le train
tree_full <- rpart(DIFF ~ ., data = train, method = "class", cp = 0.01)

# Prédiction sur le test
pred_test <- predict(tree_full, test, type = "class")


submission <- data.frame(
  id   = seq_len(nrow(test)), 
  DIFF = pred_test
)

# Export CSV
write.csv(submission, "submission.csv", row.names = FALSE)
cat("Fichier 'submission.csv' créé dans :", getwd(), "\n")

table(pred_test)

#Courbe Roc et AUC
if (!requireNamespace("ROCR", quietly = TRUE)) install.packages("ROCR", repos = "https://cloud.r-project.org")
library(ROCR)

pred <- prediction(p1, valid_set$DIFF)
perf <- performance(pred, "tpr", "fpr")
plot(perf, main = "ROC - Arbre (ROCR)")
abline(a = 0, b = 1, lty = 2)

auc_rocr <- performance(pred, "auc")@y.values[[1]]
cat("AUC (ROCR) =", auc_rocr, "\n")











