library(dplyr)
library(caret)
library(pROC)

set.seed(123) 

x=read.table("C:/Users/antoc/OneDrive/Bureau/m-1-miashs-1-2-journee-data-science/farms_train.csv",sep=",", dec=".",header=T)

x <- mutate(x, DIFF = as.factor(DIFF))

# Création des indices pour 70% apprentissage et 30% test
train_index <- createDataPartition(x$DIFF, p = 0.7, list = FALSE)

# Jeu d'apprentissage
train_data <- x[train_index, ]

# Jeu de test
test_data <- x[-train_index, ]

# Vérification dimensions
dim(train_data)
dim(test_data)

modele <- glm(DIFF ~ R2 + R7 + R8 + R17 + R22 + R32,
              data = train_data,
              family = binomial)

summary(modele)

# Test des varibales discriminantes

modele_reduit <- glm(DIFF ~ R2 + R7 + R32,
                     data = train_data, family = binomial)

AIC(modele, modele_reduit)

# AIC du modele complet est inferieur à celui du modele réduit avec les variables
# discriminantes, R2, R7, R32
# Le modele complet est donc préféré et gardé

# Probabilités prédites sur le jeu de test
proba_test <- predict(modele, newdata = test_data, type = "response")

# Classes prédites avec seuil 0.5
classe_pred_test <- ifelse(proba_test > 0.5, 1, 0)

# Matrice de confusion
confusionMatrix(
  factor(classe_pred_test, levels = c(0,1)),
  factor(test_data$DIFF,   levels = c(0,1))
)

# Calcul de l'AUC
roc_obj_test <- roc(test_data$DIFF, proba_test)
auc(roc_obj)

# Courbe ROC
plot(roc_obj_test, col = "blue", lwd = 2, main = "ROC sur données de test")
abline(a=0, b=1, lty=2, col="red")
text(0.6, 0.4, paste("AUC =", round(auc(roc_obj_test), 3)), col="blue")

# Jeu de données test
data_test=read.table("C:/Users/antoc/OneDrive/Bureau/m-1-miashs-1-2-journee-data-science/farms_test.csv",
                              sep = ",", dec = ".", header = T)

# Prédictions avec le modèle entraîné sur le 70% train
proba_test_final <- predict(modele, newdata = data_test, type = "response")

classe_pred_test_final <- ifelse(proba_test_final > 0.5, 1, 0)

resultats <- data.frame(data_test, 
                        DIFF = classe_pred_test_final)

head(resultats)
write.csv(resultats, "C:/Users/antoc/OneDrive/Bureau/m-1-miashs-1-2-journee-data-science/predictions_farms_test.csv", row.names = FALSE)



