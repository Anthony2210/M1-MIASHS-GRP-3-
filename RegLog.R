library(dplyr)

x=read.table("C:/Users/antoc/OneDrive/Bureau/m-1-miashs-1-2-journee-data-science/farms_train.csv",sep=",", dec=".",header=T)
dim(x)
head(x)
glimpse(x)

x <- mutate(x, DIFF = as.factor(DIFF))

modele <- glm(DIFF ~ R2 + R7 + R8 + R17 + R22 + R32,
              data = x,
              family = binomial)

summary(modele)

predict(modele, type = "response")

pred_class <- ifelse(predict(modele, type="response") > 0.5, 1, 0)
table(Prediction = pred_class, Réel = x$DIFF)

library(caret)   
library(pROC)    

proba <- predict(modele, type = "response")

# Classes prédites avec seuil 0.5
classe_pred <- ifelse(proba > 0.5, 1, 0)

# Matrice de confusion
confusionMatrix(
  factor(classe_pred, levels = c(0,1)),
  factor(x$DIFF,       levels = c(0,1))
)

# Calcul de l'AUC
roc_obj <- roc(x$DIFF, proba)
auc(roc_obj)

# Courbe ROC
plot(roc_obj, col = "blue", lwd = 2, main = "Courbe ROC")
abline(a=0, b=1, lty=2, col="red")  # ligne aléatoire
text(0.6, 0.4, paste("AUC =", round(auc(roc_obj), 3)), col="blue")
