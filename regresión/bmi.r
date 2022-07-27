data = read.table("diabetes2.csv", header = TRUE, sep = ',')

# se elimina la columna 'X' que es simplemente un id
data$X = NULL

# se transforma la columna 'Outcome' en factor
data$Outcome = as.factor(data$Outcome)

# scatterplots iniciales
library(GGally)
ggpairs(data)

# se eliminan las observaciones para las que 'Glucose' == 0
data = data[data[, 'Glucose'] > 0,]

# se eliminan las observaciones para las que 'BloodPressure' == 0
data = data[data[, 'BloodPressure'] > 0,]

# hay un outlier muy claro en SkinThickness
outlier = which(data$SkinThickness == max(data$SkinThickness))
data = data[-outlier,]

# scatterplots 
ggpairs(data)


# regresión múltiple para predecir la variable BMI
reg = lm(BMI~Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+DiabetesPedigreeFunction+Age+Outcome, data=data)
summary(reg)


# esperanza del BMI de una mujer con:
# - 2 embarazos
# - una concentración de glucosa de 100
# - una presión sistólica de 70
# - un valor de piel de triceps de 20
# - no tiene diabetes
# - un valor de la función pedigree de 0.24
# - 30 años de edad

reg_without_insulin = lm(BMI~Pregnancies+Glucose+BloodPressure+SkinThickness+DiabetesPedigreeFunction+Age+Outcome, data=data)

df = data.frame(Pregnancies=2, Glucose=100, BloodPressure=70, SkinThickness=20, Insulin=0, DiabetesPedigreeFunction=0.24, Age=30, Outcome=0)
df$Outcome = as.factor(df$Outcome)

predict(reg_without_insulin, df)


# intervalo de confianza y de predicción de nivel 0.95 para la estimación hallada
reg_without_insulin = lm(BMI~Pregnancies+Glucose+BloodPressure+SkinThickness+DiabetesPedigreeFunction+Age+Outcome, data=data)

df = data.frame(Intercept = 1, Pregnancies=2, Glucose=100, BloodPressure=70, SkinThickness=20, DiabetesPedigreeFunction=0.24, Age=30, Outcome=0)
df$Outcome = as.factor(df$Outcome)

predict(reg_without_insulin, newdata = df, interval = 'confidence', level = 0.95)
predict(reg_without_insulin, newdata = df, interval = 'prediction', level = 0.95)


# selección de modelos
library("bestglm")

reg = lm(BMI~Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+DiabetesPedigreeFunction+Age+Outcome, data=data)

Xmodelo = model.matrix(reg)
Y = data$BMI
Xy = data.frame(Xmodelo[, -1], Y)

set.seed(13)
res.bestglm = bestglm(Xy = Xy, IC = 'CV', method = 'exhaustive')

res.bestglm$Subsets

# modelo final
new_reg = lm(BMI~BloodPressure+SkinThickness+Pregnancies+Outcome, data=data)
summary(new_reg)


# análisis de supuestos
residuos = rstandard(new_reg)
mean(residuos)


plot(new_reg$fitted.values, residuos, pch=20, xlab='Valor ajustado', ylab='Residuos estandarizados', col='royalblue4')
title('Gráfico de residuos estandarizados en función de valores ajustados')
abline(0:9,rep(0, 10), col='red', lwd=2)

qqnorm(residuos, pch=20, col='royalblue4')
qqline(residuos, lwd=2, col='brown4')

# como no se cumple el supuesto de normalidad se realiza una transformación box-cox
library(MASS)
bc = boxcox(BMI~BloodPressure+SkinThickness+Pregnancies+Outcome, data=data)

lambda = bc$x[which.max(bc$y)]
nuevo_BMI = ((data$BMI ^ lambda) - 1) / lambda

bc_reg = lm(nuevo_BMI~BloodPressure+SkinThickness+Pregnancies+Outcome, data=data)
summary(bc_reg)

bc_residuos = rstandard(bc_reg)
mean(bc_residuos)

plot(bc_reg$fitted.values, bc_residuos, pch=20, xlab='Valor ajustado', ylab='Residuos estandarizados', col='royalblue4')
title('Gráfico de residuos estandarizados en función de valores ajustados')
abline(0:9,rep(0, 10), col='red', lwd=2)

qqnorm(bc_residuos, pch=20, col='royalblue4')
qqline(bc_residuos, lwd=2, col='brown4')


# estimación de la densidad del estimador para el coeficiente de regresión de 'SkinThickness'
Nboot = 1000

coeficientes = numeric()
for (i in 1:Nboot){
  nueva_muestra = sample(data$SkinThickness, length(data$SkinThickness), replace = TRUE)
  
  df = data.frame('BMI' = data$BMI, 'SkinThickness' = nueva_muestra)
  
  r = lm(BMI~SkinThickness, data=df)
  coeficientes[i] = r$coefficients[2]
}

hist(coeficientes, col='magenta4', xlab='Coeficientes', ylab='Frecuencia')
