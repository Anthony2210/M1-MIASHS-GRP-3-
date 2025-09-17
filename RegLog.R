library(dplyr)
library(caret)
library(pROC)

set.seed(123) 

x=read.table("C:/Users/antoc/OneDrive/Bureau/m-1-miashs-1-2-journee-data-science/farms_train.csv",sep=",", dec=".",header=T)

x <- mutate(x, DIFF = as.factor(DIFF))

# Création des indices pour 75% apprentissage et 25% test
train_index <- createDataPartition(x$DIFF, p = 0.75, list = FALSE)

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

# TEST de modèles :
# Test des variables discriminantes (trouvé graphiquement via boxplot)

modele_reduit <- glm(DIFF ~ R2 + R7 + R32,
                     data = train_data, family = binomial)

AIC(modele, modele_reduit)

# AIC du modele complet est inferieur à celui du modele réduit avec les variables
# discriminantes, R2, R7, R32
# Le modele complet est donc préféré et gardé

# Test automatique : 

modele_step <- step(modele, direction = "both")
summary(modele_step)

# AIC de ce modele est inferieur à celui du modele complet
# Ce modele est donc préféré et gardé

modele_inter <- glm(DIFF ~ R2*R32 + R17, data = train_data, family = binomial)
AIC(modele_step, modele_inter)
summary(modele_inter)

# AIC de ce modele est inferieur à celui du modele par step
# Ce modele est donc préféré et gardé

modele_poly <- glm(DIFF ~ poly(R2,2) + poly(R32,2) + R17,
                   data = train_data, family = binomial)
AIC(modele_inter, modele_poly)

# AIC de ce modele est supérieur à celui du modele précédent
# Ce modele n'est donc pas gardé

# Probabilités prédites sur le jeu de test
proba_test <- predict(modele_inter, newdata = test_data, type = "response")

# Calcul de l'AUC
roc_obj_test <- roc(test_data$DIFF, proba_test)

# Calcul du seuil optimal
seuil_opt <- coords(roc_obj_test, "best", ret = "threshold")
seuil_opt <- as.numeric(seuil_opt)

# Classes prédites avec seuil 
classe_pred_test <- ifelse(proba_test > seuil_opt, 1, 0)

# Matrice de confusion
confusionMatrix(
  factor(classe_pred_test, levels = c(0,1)),
  factor(test_data$DIFF,   levels = c(0,1))
)

auc(roc_obj_test)

# Courbe ROC
plot(roc_obj_test, col = "blue", lwd = 2, main = "ROC sur données de test")
abline(a=0, b=1, lty=2, col="red")
text(0.6, 0.4, paste("AUC =", round(auc(roc_obj_test), 3)), col="blue")

# Jeu de données test
data_test=read.table("C:/Users/antoc/OneDrive/Bureau/m-1-miashs-1-2-journee-data-science/farms_test.csv",
                              sep = ",", dec = ".", header = T)

# Prédictions avec le modèle entraîné sur le 70% train
proba_test_final <- predict(modele_inter, newdata = data_test, type = "response")

classe_pred_test_final <- ifelse(proba_test_final > seuil_opt, 1, 0)

resultats <- data.frame(
  ID   = 1:nrow(data_test),
  DIFF = classe_pred_test_final
)

head(resultats)
write.csv(resultats, "C:/Users/antoc/OneDrive/Bureau/m-1-miashs-1-2-journee-data-science/predictions_farms_test.csv", row.names = FALSE)
