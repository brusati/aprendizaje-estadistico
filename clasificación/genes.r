x = read.delim("colonX.txt", sep = ' ', header = FALSE)

colores = c('skyblue', 'chocolate')

set.seed(22)





# normalización de los datos
medians = c()
for (i in 1:nrow(x)){
  medians[i] = median(as.matrix(x[i,]))
}
hist(medians, col='purple')

x = log(x)

log_medians = c()
for (i in 1:nrow(x)){
  log_medians[i] = median(as.matrix(x[i,]))
}
hist(log_medians, col='purple')





# preparación de la variable objetivo
classes = read.delim("colonT.txt", header = FALSE)

y = c()
for (i in 1:length(classes$V1)){
  if (classes$V1[i] > 0) y[i] = 0
  else y[i] = 1
}





# separación en train y test
posiciones = sample(1:62, size=18, replace = FALSE)

x_test = t(x)[posiciones,]
x_train = t(x)[-posiciones,]

y_test = y[posiciones]
y_train = y[-posiciones]





data = as.data.frame(cbind(x_train, y_train))

# árboles comunes
library(rpart)
library(rpart.plot)

set.seed(22)

arbol = rpart(y_train~., data=data, method='class')
arbol
summary(arbol)

rpart.plot(arbol)

predichos_arbol = predict(arbol, newdata=as.data.frame(x_test))

error_arbol = mean(na.omit((predichos_arbol-as.numeric(y_test))^2))
error_arbol

library(ROCit)
probas_arbol = predict(arbol, newdata=as.data.frame(x_test), type='prob')[, 2]
ROC_arbol = rocit(score=probas_arbol, class=as.factor(y_test))
plot(ROC_arbol, col='chocolate')
title('ROC curve for Classification Tree')
ROC_arbol$AUC

distancia = sqrt((ROC_arbol$FPR)^2 + (ROC_arbol$TPR-1)^2)
minimo = ROC_arbol$Cutoff[which.min(distancia)]
minimo





# bagging
library(ipred)

set.seed(22)

bag = bagging(y_train~., data=data, nbagg=15)

predichos_bagging = predict(bag, newdata=as.data.frame(x_test))

error_bagging = mean(na.omit((predichos_bagging-as.numeric(y_test))^2))
error_bagging
# se reduce significativamente el error

library(ROCit)
probas_bagging = predict(bag, newdata=as.data.frame(x_test), type='prob')
ROC_bagging = rocit(score=probas_bagging, class=as.factor(y_test))
plot(ROC_bagging, col='chocolate')
title('ROC curve for Bagging')
ROC_bagging$AUC

distancia = sqrt((ROC_bagging$FPR)^2 + (ROC_bagging$TPR-1)^2)
minimo = ROC_bagging$Cutoff[which.min(distancia)]
minimo



# random forest
library(randomForest)

set.seed(22)

data_rf = as.data.frame(cbind(x_train, y_train))
data_rf$y_train = as.factor(data_rf$y_train)

# busco el mtry óptimo
mtrys = 30:60
l_sombrero_rf = c()

for (j in 1:length(mtrys)){
  set.seed(22)
  rf = randomForest(y_train~., data=data_rf, mtry=mtrys[j], importance=TRUE)
  predichos = predict(rf, newdata=as.data.frame(x_test))
  l_sombrero_rf[j] = mean(predichos != y_test)
}

plot(mtrys, l_sombrero_rf, pch=20, col='darkolivegreen', xlab='Number of variables randomly sampled', ylab='MSE')
title('Different number of variables randomly sampled and their errors')

mejor_mtry = mtrys[which(min(l_sombrero_rf) == l_sombrero_rf)[1]]

# rf con el mejor mtry
set.seed(22)
rf = randomForest(y_train~., data=data_rf, mtry=mejor_mtry, importance=TRUE)

predichos_rf = predict(rf, newdata=as.data.frame(x_test))

error_rf = mean(na.omit((as.numeric(predichos_rf)-as.numeric(y_test)-1)^2))
error_rf

library(ROCit)
probas_rf = predict(rf, newdata=as.data.frame(x_test), type='prob')[, 2]
ROC_rf = rocit(score=probas_rf, class=as.factor(y_test))
plot(ROC_rf, col='chocolate')
title('ROC curve for Random Forest')
ROC_rf$AUC

importance(rf)
varImpPlot(rf)

distancia = sqrt((ROC_rf$FPR)^2 + (ROC_rf$TPR-1)^2)
minimo = ROC_rf$Cutoff[which.min(distancia)]
minimo





# boosting
library(gbm)
set.seed(22)

# busco el n.trees óptimo
ntrees = seq(500, 10000, 500)
l_sombrero_boosting = c()

for (j in 1:length(ntrees)){
  set.seed(22)
  boosting = gbm(y_train~., data=data, distribution='bernoulli', n.trees=ntrees[j])
  probas = predict(boosting, newdata=as.data.frame(x_test), n.trees=ntrees[j], type='response')
  predichos = ifelse(probas < 0.5, 0, 1)
  l_sombrero_boosting[j] = mean(predichos != y_test)
}

plot(ntrees, l_sombrero_boosting, pch=20, col='darkolivegreen', xlab='Number of trees', ylab='MSE')
title('Different number of trees and their errors')

mejor_ntree = ntrees[which(min(l_sombrero_boosting) == l_sombrero_boosting)[1]]

# boosting con el n.trees óptimo
set.seed(22)

boosting = gbm(y_train~., data=data, distribution='bernoulli', n.trees=mejor_ntree)

probas_boosting = predict(boosting, newdata=as.data.frame(x_test), n.trees=mejor_ntree, type='response')
predichos_boosting = ifelse(probas_boosting < 0.5, 0, 1)

error_boosting = mean(na.omit((predichos_boosting-as.numeric(y_test))^2))
error_boosting

library(ROCit)
ROC_boosting = rocit(score=probas_boosting, class=as.factor(y_test))
plot(ROC_boosting, col='chocolate')
title('ROC curve for Boosting')
ROC_boosting$AUC

distancia = sqrt((ROC_boosting$FPR)^2 + (ROC_boosting$TPR-1)^2)
minimo = ROC_boosting$Cutoff[which.min(distancia)]
minimo





# nearest shrunken centroids
# documentación de la función: https://search.r-project.org/CRAN/refmans/pamr/html/pamr.train.html
library(cluster)
library(survival)
library(pamr)

input_data_train = list(x=data.matrix(t(x_train)), y=factor(as.integer(y_train)), genenames=as.character(1:2000))
input_data_test = list(x=data.matrix(t(x_test)), y=factor(as.integer(y_test)), genenames=as.character(1:2000))

# busco el n.threshold óptimo
nthresholds = seq(1, 10, 1)
l_sombrero_nsc = c()

for (j in 1:length(nthresholds)){
  set.seed(22)
  nsc = pamr.train(input_data_train, n.threshold=nthresholds[j])
  predichos = pamr.predict(nsc, newx=input_data_test$x, threshold=nthresholds[j])
  l_sombrero_nsc[j] = mean(predichos != y_test)
}

plot(nthresholds, l_sombrero_nsc, pch=20, col='darkolivegreen', xlab='Thresholds', ylab='MSE')
title('Different thresholds and their errors')

mejor_nthreshold = nthresholds[which(min(l_sombrero_nsc) == l_sombrero_nsc)[1]]

# nearest shrunken centroids con el n.thresholds óptimo
set.seed(22)
nsc = pamr.train(input_data_train, n.threshold=mejor_nthreshold)

predichos_nsc = pamr.predict(nsc, newx=input_data_test$x, threshold=mejor_nthreshold)

error_nsc = mean(na.omit((as.numeric(predichos_nsc)-1-as.numeric(y_test))^2))
error_nsc

library(ROCit)
probas_nsc = pamr.predict(nsc, newx=input_data_test$x, threshold=mejor_nthreshold, type='posterior')[, 2]
ROC_nsc = rocit(score=probas_nsc, class=as.factor(y_test))
plot(ROC_nsc, col='chocolate')
title('ROC curve for Nearest Shrunken Centroids')
ROC_nsc$AUC

# genes más importantes
input_data_genes = list(x=data.matrix(t(x_train)), y=factor(as.integer(y_train)), geneid=as.character(1:2000))
genes = pamr.listgenes(nsc, input_data_genes, threshold=mejor_nthreshold)[, 1]
genes = as.numeric(genes)
genes

distancia = sqrt((ROC_nsc$FPR)^2 + (ROC_nsc$TPR-1)^2)
minimo = ROC_nsc$Cutoff[which.min(distancia)]
minimo



data = as.data.frame(cbind(x_train[, genes], y_train))



# knn
library(class)

# busco el k óptimo
k = 1:25
l_sombrero_knn = c()

set.seed(22)

for (j in 1:length(k)){
  predichos = knn(train=x_train[, genes], cl=as.vector(factor(y_train)), k=k[j], test=x_test[, genes])
  l_sombrero_knn[j] = mean(predichos != y_test)
}

plot(k, l_sombrero_knn, pch=20, col='darkolivegreen', xlab='Number of neighbors', ylab='MSE')
title('Different number of neighbors and their errors')

mejor_k = k[which(min(l_sombrero_knn) == l_sombrero_knn)[1]]


# knn usando el k óptimo
set.seed(22)
predichos_knn = knn(train=x_train[, genes], cl=as.vector(factor(y_train)), k=mejor_k, test=x_test[, genes], prob=TRUE)

error_knn = mean(na.omit((as.numeric(predichos_knn)-as.numeric(y_test)-1)^2))
error_knn

library(ROCit)
probas_knn = attr(predichos_knn, 'prob')
ROC_knn = rocit(score=probas_knn, class=as.factor(y_test))
plot(ROC_knn, col='chocolate')
title('ROC curve for KNN')
ROC_knn$AUC

distancia = sqrt((ROC_knn$FPR)^2 + (ROC_knn$TPR-1)^2)
minimo = ROC_knn$Cutoff[which.min(distancia)]
minimo





# regresión logística
set.seed(22)
logistica = glm(y_train~., data=data, family=binomial)
summary(logistica)

probas_lr = predict(logistica, new=as.data.frame(x_test[, genes]), type='response')
predichos_lr = ifelse(probas_lr < 0.5, 0, 1)

error_lr = mean(na.omit((as.numeric(predichos_lr)-as.numeric(y_test))^2))
error_lr

library(ROCit)
ROC_lr = rocit(score=probas_lr, class=as.factor(y_test))
plot(ROC_lr, col='chocolate')
title('ROC curve for Logistic Regression')
ROC_lr$AUC

distancia = sqrt((ROC_lr$FPR)^2 + (ROC_lr$TPR-1)^2)
minimo = ROC_lr$Cutoff[which.min(distancia)]
minimo



# análisis discriminante
set.seed(22)

enfermos_x_train = x_train[which(y_train == 1), genes]
sanos_x_train = x_train[which(y_train == 0), genes]
enfermos_x_test = x_test[which(y_test == 1), genes]
sanos_x_test = x_test[which(y_test == 0), genes]

# verificamos normalidad
qqnorm((enfermos_x_train - mean(enfermos_x_train)) / sd(enfermos_x_train))
abline(0, 1, col='red')

qqnorm((sanos_x_train - mean(sanos_x_train)) / sd(sanos_x_train))
abline(0, 1, col='red')


enfermos_y_train = rep(1, nrow(enfermos_x_train))
sanos_y_train = rep(0, nrow(sanos_x_train))
enfermos_y_test = rep(1, nrow(enfermos_x_test))
sanos_y_test = rep(0, nrow(sanos_x_test))

library(MASS)
library(klaR)

# lda
y_lda_train = c(as.factor(enfermos_y_train), as.factor(sanos_y_train))
x_lda_train = rbind(enfermos_x_train, sanos_x_train)
y_lda_test = c(as.factor(enfermos_y_test), as.factor(sanos_y_test))
x_lda_test = rbind(enfermos_x_test, sanos_x_test)

lda = lda(x_lda_train, y_lda_train)
plot(lda)

predichos = predict(lda, x_lda_test)
probas_lda = predichos$posterior[, 1]
predichos_lda = predichos$class

error_lda = mean(na.omit((as.numeric(predichos_lda)-1-as.numeric(y_test))^2))
error_lda

library(ROCit)
ROC_lda = rocit(score=probas_lda, class=as.factor(y_test))
plot(ROC_lda, col='chocolate')
title('ROC curve for LDA')
ROC_lda$AUC

distancia = sqrt((ROC_lda$FPR)^2 + (ROC_lda$TPR-1)^2)
minimo = ROC_lda$Cutoff[which.min(distancia)]
minimo




# qda
y_qda_train = c(as.factor(enfermos_y_train), as.factor(sanos_y_train))
x_qda_train = rbind(enfermos_x_train[, 1:12], sanos_x_train[, 1:12])
y_qda_test = c(as.factor(enfermos_y_test), as.factor(sanos_y_test))
x_qda_test = rbind(enfermos_x_test[, 1:12], sanos_x_test[, 1:12])

qda = qda(x_qda_train, y_qda_train)

predichos = predict(qda, as.data.frame(x_qda_test))
probas_qda = predichos$posterior[, 1]
predichos_qda = predichos$class

error_qda = mean(na.omit((as.numeric(predichos_qda)-1-as.numeric(y_test))^2))
error_qda

library(ROCit)
ROC_qda = rocit(score=probas_qda, class=as.factor(y_test))
plot(ROC_qda, col='chocolate')
title('ROC curve for QDA')
ROC_qda$AUC

distancia = sqrt((ROC_qda$FPR)^2 + (ROC_qda$TPR-1)^2)
minimo = ROC_qda$Cutoff[which.min(distancia)]
minimo
