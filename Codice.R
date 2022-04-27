###############################################################################
#                                                                             #  
#  Progetto ML - appello febbraio 2021 - Gagianesi Matteo e Talamona Stefano  #
#                                                                             #
###############################################################################

# INDICE:
#         1 - Librerie utilizzate
#         2 - Preprocessing
#         3 - Analisi delle distribuzioni
#         4 - PCA
#         5 - SVM
#         6 - Neural Network
#         7 - Naive Bayes
#         8 - Decision Tree
#         9 - Model Evaluation

##########################   Librerie utilizzate   #############################

# # Istallazione librerie
# install.packages(c("rstudioapi", "fitdistrplus", "FactoMineR", "factoextra",
#                    "e1071", "ggplot2", "caret", "nnet", "naivebayes", "rpart",
#                    "rattle", "randomForest", "ROCR", "pROC"))

# Preprocessing e Analisi delle distribuzioni
library(rstudioapi)
library(fitdistrplus)
library(ggplot2)
library(caret)
# PCA
library(FactoMineR)
library(factoextra)
# SVM
library(e1071)
# Neural Network
library(nnet)
# Naive Bayes
library(naivebayes)
# Decision Tree
library(rpart)
library(rattle)
library(randomForest)
# Model Evalutation
library(ROCR)
library(pROC)



##############################   Preprocessing   ###############################


# Viene settata come working directory il path attuale
current_path = getActiveDocumentContext()$path
setwd(dirname(current_path))
# Loading del dataset
kepler <- read.csv("kepler.csv", stringsAsFactors = T, sep = ',', header = TRUE)


# ELIMINAZIONE DELLE FEATURE SCARTATE
# ID e nominativi
kepler$rowid <- NULL
kepler$kepid <- NULL
kepler$kepoi_name <- NULL
kepler$kepler_name <- NULL
kepler$koi_pdisposition <- NULL
kepler$koi_tce_delivname <- NULL
# Lilello di confidenza dei test (NASA) per stabilire la natura delle osservazioni
kepler$koi_score <- NULL
# Colonne vuote
kepler$koi_teq_err1 <- NULL
kepler$koi_teq_err2 <- NULL


# Per ogni feature del dataset che è stata mantenuta, stampa numero e
# percentuale dei valori mancanti
for (i in 1 : ncol(kepler)){
  print(paste(colnames(kepler[i]), "|   Valori mancanti: ", sum(is.na(kepler[i])),
              "|   % di valori mancanti: ", sum(is.na(kepler[i]))/nrow(kepler)*100))
}
# Elimina le istanze con valori mancanti
kepler <- na.omit(kepler)


# Numero di istanze CONFIREMED, CANDIDATE e FALSE POSITIVE
confirmed <- 0
false_positive <- 0
candidate <- 0
for(i in 1 : nrow(kepler)){
  if(kepler$koi_disposition[i] == 'CONFIRMED')
    confirmed = confirmed + 1
  else if(kepler$koi_disposition[i] == 'CANDIDATE')
    candidate = candidate + 1
  else if(kepler$koi_disposition[i] == 'FALSE POSITIVE')
    false_positive = false_positive + 1
}
cat(sprintf(' Numero di istanze "CANDIDATE": \'%i', candidate))
cat(sprintf(' Numero di istanze "CONFIRMED": \'%i', confirmed))
cat(sprintf(' Numero di istanze "FALSE POSITIVE": \'%i', false_positive))


# Vengono tenute solo le istanze con label "CONFIRMED" e "FALSE POSITIVE"
kepler <- kepler[kepler$koi_disposition != "CANDIDATE", ]
print(paste(" Rimosse tutte le", candidate, "istanze con label 'CANDIDATE' "))
kepler$koi_disposition <- factor(kepler$koi_disposition) # Rimosso il level "CANDIDATE"


# Ricerca di variabili in forte correlazione reciproca
M <- cor(kepler[, 2 : ncol(kepler)])
highlyCorrelated <- findCorrelation(M, cutoff = 0.7, names = TRUE, verbose = TRUE)
highlyCorrelated
# Prima di eliminare le variabili fortemente correlate viene creata una copia
# del dataset, al fine di potervi eseguire la PCA e l'analisi delle
# distribuzioni delle feature
kepler.og <- kepler
# Vengono eliminate dal dataset le variabili fortemente correlate tra loro,
# ovvero: koi_duration_err1, koi_period_err1, koi_srad, koi_time0bk_err1, 
#         koi_srad_err2, koi_insol, koi_insol_err1, koi_impact_err2,
#         koi_prad, koi_prad_err1, koi_steff_err1, koi_depth_err2
kepler <- kepler[-c(7, 10, 14, 16, 20 : 22, 25, 26, 31, 36, 38)]


# Viene effettuato lo scaling del dataset
kepler[, 2 : ncol(kepler)] <- scale(kepler[, 2 : ncol(kepler)],
                                     center = TRUE, scale = TRUE)

# Il dataset viene preventivamente diviso in training (70%) e test (30%)
ind = sample(2, nrow(kepler), replace = TRUE, prob = c(0.7, 0.3))
trainset = kepler[ind == 1, ]
testset = kepler[ind == 2, ]



######################   Analisi delle distribuzioni   #########################


# Distribuzione delle label "FALSE POSITIVE" e "CONFIRMED"
plot(kepler.og$koi_disposition, 
     main = "Frequenza istanze confermate e falsi positivi", xlab = "Label Target", 
     col = c("#00CADF", "#FF624D"))


# Distribuzione delle quattro flag nelle istanze con label "FALSE POSITIVE"
par(mfrow = c(2, 2))

# Distribuzione della flag koi_fpflag_nt
plot(as.factor(kepler.og$koi_fpflag_nt[kepler.og$koi_disposition == "FALSE POSITIVE"]), 
     main = NULL, xlab ="koi_fpflag_nt", col = c("#00CADF", "#FF624D"))

# Distribuzione della flag koi_fpflag_ss
plot(as.factor(kepler.og$koi_fpflag_ss[kepler.og$koi_disposition == "FALSE POSITIVE"]), 
     main = NULL, xlab = "koi_fpflag_ss",  col = c("#00CADF", "#FF624D"))

# Distribuzione della flag koi_fpflag_co
plot(as.factor(kepler.og$koi_fpflag_co[kepler.og$koi_disposition == "FALSE POSITIVE"]), 
     main = NULL, xlab = "koi_fpflag_co", col = c("#00CADF", "#FF624D"))

# Distribuzione della flag koi_fpflag_ec
plot(as.factor(kepler.og$koi_fpflag_ec[kepler.og$koi_disposition == "FALSE POSITIVE"]), 
     main = NULL, xlab = "koi_fpflag_ec", col = c("#00CADF", "#FF624D"))
title(main = "Distribuzione nelle istanze FALSE POSITIVE", line = -2,
      cex.main = 2, outer = TRUE)


# Distribuzione delle quattro flag nelle istanze con label "CONFIRMED"
par(mfrow = c(2, 2))

# Distribuzione della flag koi_fpflag_nt
plot(as.factor(kepler.og$koi_fpflag_nt[kepler.og$koi_disposition == "CONFIRMED"]), 
     main = NULL, xlab = "koi_fpflag_nt", col = c("#00CADF", "#FF624D"))

# Distribuzione della flag koi_fpflag_ss
plot(as.factor(kepler.og$koi_fpflag_ss[kepler.og$koi_disposition == "CONFIRMED"]), 
     main = NULL, xlab = "koi_fpflag_ss",  col = c("#00CADF", "#FF624D"))

# Distribuzione della flag koi_fpflag_co
plot(as.factor(kepler.og$koi_fpflag_co[kepler.og$koi_disposition == "CONFIRMED"]), 
     main = NULL, xlab = "koi_fpflag_co", col = c("#00CADF", "#FF624D"))

# Distribuzione della flag koi_fpflag_ec
plot(as.factor(kepler.og$koi_fpflag_ec[kepler.og$koi_disposition == "CONFIRMED"]), 
     main = NULL, xlab = "koi_fpflag_ec", col = c("#00CADF", "#FF624D"))
title(main = "Distribuzione nelle istanze CONFIRMED", line = -2,
      cex.main = 2, outer = TRUE)


# Il layout per i plot successivi viene reimpostato ad un solo grafico per plot
par(mfrow = c(1, 1))


# Distribuzione dei valori di koi_tce_plnt_num
plot(as.factor(kepler.og$koi_tce_plnt_num),
     main = "Numero di Threshold Crossing Events",
     xlab = "TCEs", col = c("#00CADF", "#FF8C00", "#9A00FF", "#0000FF",
                            "#FFFF00", "#00FF00", "#FF47FF", "#FF624D"))


# koi_period
q = quantile(kepler.og$koi_period)
hist(kepler.og$koi_period, main = "Distribuzione koi_period",
     xlab = "koi_period", freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_period, main = "Distribuzione koi_period",
        xlab = "koi_period")
hist(kepler.og$koi_period, main = "Distribuzione koi_period",
     xlab = "koi_period", freq = FALSE)
lines(density(kepler.og$koi_period), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_period, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_period, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_period, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_period, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
    fit.mle <- fit.mle.lnorm
} else
    fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_period_err2
q = quantile(kepler.og$koi_period_err2)
hist(kepler.og$koi_period_err2, main = "Distribuzione koi_period_err2", 
     xlab = "koi_period_err2", freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_period_err2, main = "Distribuzione koi_period_err2",
        xlab = "koi_period_err2")
hist(kepler.og$koi_period_err2, main = "Distribuzione koi_period_err2",
     xlab = "koi_period_err2", freq = FALSE)
lines(density(kepler.og$koi_period_err2), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_period_err2, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_period_err2, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_period_err2, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_period_err2, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_time0bk
q = quantile(kepler.og$koi_time0bk)
hist(kepler.og$koi_time0bk, main = "Distribuzione koi_time0bk",
     xlab = "koi_time0bk", freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_time0bk, main = "Distribuzione koi_time0bk",
        xlab = "koi_time0bk")
hist(kepler.og$koi_time0bk, main = "Distribuzione koi_time0bk",
     xlab = "koi_time0bk", freq = FALSE)
lines(density(kepler.og$koi_time0bk), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_time0bk, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_time0bk, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_time0bk, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_time0bk, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_time0bk_err2
q = quantile(kepler.og$koi_time0bk_err2)
hist(kepler.og$koi_time0bk_err2, main = "Distribuzione koi_time0bk_err2",
     xlab = "koi_time0bk_err2", freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_time0bk_err2, main = "Distribuzione koi_time0bk_err2",
        xlab = "koi_time0bk_err2")
hist(kepler.og$koi_time0bk_err2, main = "Distribuzione koi_time0bk_err2",
     xlab = "koi_time0bk_err2", freq = FALSE)
lines(density(kepler.og$koi_time0bk_err2), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_time0bk_err2, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_time0bk_err2, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_time0bk_err2, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_time0bk_err2, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_impact
q = quantile(kepler.og$koi_impact)
hist(kepler.og$koi_impact, main = "Distribuzione koi_impact",
     xlab = "koi_impact", freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_impact, main = "Distribuzione koi_impact",
        xlab = "koi_impact")
hist(kepler.og$koi_impact, main = "Distribuzione koi_impact",
     xlab = "koi_impact", freq = FALSE)
lines(density(kepler.og$koi_impact), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_impact, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_impact, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_impact, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_impact, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_impact_err1
q = quantile(kepler.og$koi_impact_err1)
hist(kepler.og$koi_impact_err1, main = "Distribuzione koi_impact_err1",
     xlab = "koi_impact_err1", freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_impact_err1, main = "Distribuzione koi_impact_err1",
        xlab = "koi_impact_err1")
hist(kepler.og$koi_impact_err1, main = "Distribuzione koi_impact_err1",
     xlab = "koi_impact_err1", freq = FALSE)
lines(density(kepler.og$koi_impact_err1), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_impact_err1, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_impact_err1, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_impact_err1, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_impact_err1, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_duration
q = quantile(kepler.og$koi_duration)
hist(kepler.og$koi_duration, main = "Distribuzione koi_duration",
     xlab = "koi_duration", freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_duration, main = "Distribuzione koi_duration",
        xlab = "koi_duration")
hist(kepler.og$koi_duration, main = "Distribuzione koi_duration",
     xlab = "koi_duration", freq = FALSE)
lines(density(kepler.og$koi_duration), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_duration, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_duration, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_duration, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_duration, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_duration_err2
q = quantile(kepler.og$koi_duration_err2)
hist(kepler.og$koi_duration_err2, main = "Distribuzione koi_duration_err2",
     xlab = "koi_duration_err2", freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_duration_err2, main = "Distribuzione koi_duration_err2",
        xlab = "koi_duration_err2")
hist(kepler.og$koi_duration_err2, main = "Distribuzione koi_duration_err2",
     xlab = "koi_duration_err2", freq = FALSE)
lines(density(kepler.og$koi_duration_err2), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_duration_err2, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_duration_err2, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_duration_err2, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_duration_err2, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_depth
q = quantile(kepler.og$koi_depth)
hist(kepler.og$koi_depth, main = "Distribuzione koi_depth", xlab = "koi_depth",
     freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_depth, main = "Distribuzione koi_depth", xlab = "koi_depth")
hist(kepler.og$koi_depth, main = "Distribuzione koi_depth", xlab = "koi_depth",
     freq = FALSE)
lines(density(kepler.og$koi_depth), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_depth, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_depth, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_depth, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_depth, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_depth_err1
q = quantile(kepler.og$koi_depth_err1)
hist(kepler.og$koi_depth_err1, main = "Distribuzione koi_depth_err1",
     xlab = "koi_depth_err1", freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_depth_err1, main = "Distribuzione koi_depth_err1",
        xlab = "koi_depth_err1")
hist(kepler.og$koi_depth_err1, main = "Distribuzione koi_depth_err1",
     xlab = "koi_depth_err1", freq = FALSE)
lines(density(kepler.og$koi_depth_err1), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_depth_err1, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_depth_err1, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_depth_err1, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_depth_err1, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_prad_err2
q = quantile(kepler.og$koi_prad_err2)
hist(kepler.og$koi_prad_err2, main = "Distribuzione koi_prad_err2",
     xlab = "koi_prad_err2", freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_prad_err2, main = "Distribuzione koi_prad_err2",
        xlab = "koi_prad_err2")
hist(kepler.og$koi_prad_err2, main = "Distribuzione koi_prad_err2",
     xlab = "koi_prad_err2", freq = FALSE)
lines(density(kepler.og$koi_prad_err2), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_prad_err2, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_prad_err2, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_prad_err2, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_prad_err2, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_teq
q = quantile(kepler.og$koi_teq)
hist(kepler.og$koi_teq, main = "Distribuzione koi_teq", xlab = "koi_teq",
     freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_teq, main = "Distribuzione koi_teq", xlab = "koi_teq")
hist(kepler.og$koi_teq, main = "Distribuzione koi_teq", xlab = "koi_teq",
     freq = FALSE)
lines(density(kepler.og$koi_teq), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_teq, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_teq, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_teq, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_teq, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_insol_err2
q = quantile(kepler.og$koi_insol_err2)
hist(kepler.og$koi_insol_err2, main = "Distribuzione koi_insol_err2", 
     xlab = "koi_insol_err2", freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_insol_err2, main = "Distribuzione koi_insol_err2",
        xlab = "koi_insol_err2")
hist(kepler.og$koi_insol_err2, main = "Distribuzione koi_insol_err2",
     xlab = "koi_insol_err2", freq = FALSE)
lines(density(kepler.og$koi_insol_err2), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_insol_err2, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_insol_err2, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_insol_err2, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_insol_err2, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_model_snr
q = quantile(kepler.og$koi_model_snr)
hist(kepler.og$koi_model_snr, main = "Distribuzione koi_model_snr",
     xlab = "koi_model_snr", freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_model_snr, main = "Distribuzione koi_model_snr",
        xlab = "koi_model_snr")
hist(kepler.og$koi_model_snr, main = "Distribuzione koi_model_snr",
     xlab = "koi_model_snr", freq = FALSE)
lines(density(kepler.og$koi_model_snr), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_model_snr, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_model_snr, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_model_snr, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_model_snr, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_steff
q = quantile(kepler.og$koi_steff)
hist(kepler.og$koi_steff, main = "Distribuzione koi_steff", xlab = "koi_steff",
     freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_steff, main = "Distribuzione koi_steff", xlab = "koi_steff")
hist(kepler.og$koi_steff, main = "Distribuzione koi_steff",
     xlab = "koi_steff", freq = FALSE)
lines(density(kepler.og$koi_steff), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_steff, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_steff, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_steff, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_steff, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_steff_err2
q = quantile(kepler.og$koi_steff_err2)
hist(kepler.og$koi_steff_err2, main = "Distribuzione koi_steff_err2",
     xlab = "koi_steff_err2", freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_steff_err2, main = "Distribuzione koi_steff_err2",
        xlab = "koi_steff_err2")
hist(kepler.og$koi_steff_err2, main = "Distribuzione koi_steff_err2",
     xlab = "koi_steff_err2", freq = FALSE)
lines(density(kepler.og$koi_steff_err2), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_steff_err2, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_steff_err2, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_steff_err2, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_steff_err2, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_slogg
q = quantile(kepler.og$koi_slogg)
hist(kepler.og$koi_slogg, main = "Distribuzione koi_slogg", xlab = "koi_slogg",
     freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_slogg, main = "Distribuzione koi_slogg", xlab = "koi_slogg")
hist(kepler.og$koi_slogg, main = "Distribuzione koi_slogg", xlab = "koi_slogg",
     freq = FALSE)
lines(density(kepler.og$koi_slogg), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_slogg, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_slogg, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_slogg, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_slogg, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_slogg_err1
q = quantile(kepler.og$koi_slogg_err1)
hist(kepler.og$koi_slogg_err1, main = "Distribuzione koi_slogg_err1",
     xlab = "koi_slogg_err1", freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_slogg_err1, main = "Distribuzione koi_slogg_err1",
        xlab = "koi_slogg_err1")
hist(kepler.og$koi_slogg_err1, main = "Distribuzione koi_slogg_err1",
     xlab = "koi_slogg_err1", freq = FALSE)
lines(density(kepler.og$koi_slogg_err1), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_slogg_err1, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_slogg_err1, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_slogg_err1, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_slogg_err1, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_slogg_err2
q = quantile(kepler.og$koi_slogg_err2)
hist(kepler.og$koi_slogg_err2, main = "Distribuzione koi_slogg_err2",
     xlab = "koi_slogg_err2", freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_slogg_err2, main = "Distribuzione koi_slogg_err2",
        xlab = "koi_slogg_err2")
hist(kepler.og$koi_slogg_err2, main = "Distribuzione koi_slogg_err2",
     xlab = "koi_slogg_err2", freq = FALSE)
lines(density(kepler.og$koi_slogg_err2), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_slogg_err2, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_slogg_err2, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_slogg_err2, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_slogg_err2, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_srad_err1
q = quantile(kepler.og$koi_srad_err1)
hist(kepler.og$koi_srad_err1, main = "Distribuzione koi_srad_err1",
     xlab = "koi_srad_err1", freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_srad_err1, main = "Distribuzione koi_srad_err1",
        xlab = "koi_srad_err1")
hist(kepler.og$koi_srad_err1, main = "Distribuzione koi_srad_err1",
     xlab = "koi_srad_err1", freq = FALSE)
lines(density(kepler.og$koi_srad_err1), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_srad_err1, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_srad_err1, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_srad_err1, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_srad_err1, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# ra
q = quantile(kepler.og$ra)
hist(kepler.og$ra, main = "Distribuzione ra", xlab = "ra",
     freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$ra, main = "Distribuzione ra", xlab = "ra")
hist(kepler.og$ra, main = "Distribuzione ra", xlab = "ra", freq = FALSE)
lines(density(kepler.og$ra), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$ra, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$ra, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$ra, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$ra, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# dec
q = quantile(kepler.og$dec)
hist(kepler.og$dec, main = "Distribuzione dec", xlab = "dec",
     freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$dec, main = "Distribuzione dec", xlab = "dec")
hist(kepler.og$dec, main = "Distribuzione dec", xlab = "dec", freq = FALSE)
lines(density(kepler.og$dec), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$dec, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$dec, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$dec, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$dec, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)


# koi_kepmag
q = quantile(kepler.og$koi_kepmag)
hist(kepler.og$koi_kepmag, main = "Distribuzione koi_kepmag", xlab = "koi_kepmag",
     freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% 
abline(v = q[2], col = "blue", lwd = 2) # Primo quantile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Distribuzione valore mediano 50%
abline(v = q[4], col = "blue", lwd = 2) # Terzo quantile 75%
abline(v = q[5], col = "red", lwd = 2) # 100%
boxplot(kepler.og$koi_kepmag, main = "Distribuzione koi_kepmag",
        xlab = "koi_kepmag")
hist(kepler.og$koi_kepmag, main = "Distribuzione koi_kepmag",
     xlab = "koi_kepmag", freq = FALSE)
lines(density(kepler.og$koi_kepmag), col = "red")

# Viene controllata la distribuzione che più si avvicina a quella osservata
descdist(kepler.og$koi_kepmag, discrete = FALSE, boot = 1000)
fit.mle.norm <- fitdist(kepler.og$koi_kepmag, "norm", method = "mle") # NORMAL
fit.mle.lnorm <- fitdist(kepler.og$koi_kepmag, "lnorm", method = "mle") # LOGNORMAL
fit.mle.gamma <- fitdist(kepler.og$koi_kepmag, "gamma", method = "mle") # GAMMA

# Viene considerata solo quella con massimo valore di likelihood
if(fit.mle.norm$loglik == max(fit.mle.norm$loglik, fit.mle.lnorm$loglik,
                              fit.mle.gamma$loglik)){
  fit.mle <- fit.mle.norm
} else if(fit.mle.lnorm$loglik > fit.mle.gamma$loglik){
  fit.mle <- fit.mle.lnorm
} else
  fit.mle <- fit.mle.gamma
plot(fit.mle)
summary(fit.mle)



##################################   PCA   #####################################


# PCA sul dataset
kepler.og.active <- kepler.og[, 2 : ncol(kepler.og)]
pca <- PCA(kepler.og.active, scale.unit = TRUE, ncp = 20, graph = FALSE)

# Lunghezza degli autovettori
eigenvalues <- get_eigenvalue(pca)
eigenvalues

# Grafici illustrativi
fviz_eig(pca, addlabels = TRUE, ncp = ncol(kepler.og.active),
         ylim = c(0, 50)) # % di contributo alla varianza
fviz_contrib(pca, choice = "var", axes = 1 : 20) # % di contributo alle prime 20 PCs
fviz_pca_var(pca, col.var = "black") # Grafico PCA

# Correlazione e contributo delle variabili
var <- get_pca_var(pca)
var$cor # Correlazioni tra variabili
var$contrib # Contributo



##################################   SVM   #####################################


# Training dei modelli
# LINEAR
svm.model.linear = svm(formula = koi_disposition ~ ., data = trainset,
                       type = 'C', kernel = 'linear', cost = 10, scale = FALSE)
# POLYNOMIAL
svm.model.polynomial = svm(formula = koi_disposition ~ ., data=trainset,
                           type = 'C', kernel='polynomial', cost = 10, scale = FALSE)
# RADIAL
svm.model.radial = svm(formula = koi_disposition ~ ., data=trainset,
                       type = 'C', kernel='radial', cost = 10, scale = FALSE)
# SIGMOID
svm.model.sigmoid = svm(formula = koi_disposition ~ ., data = trainset,
                        type = 'C', kernel = 'sigmoid', cost = 10, scale = FALSE)

# Creazione della previsione del testset utilizzando il modello allenato
svm.prediction.linear <- predict(svm.model.linear, testset, type = "class")
svm.prediction.polynomial <- predict(svm.model.polynomial, testset, type = "class")
svm.prediction.radial <- predict(svm.model.radial, testset, type = "class")
svm.prediction.sigmoid <- predict(svm.model.sigmoid, testset, type = "class")

# Creazione CONFUSION MATRIX
svm.table.linear = table(svm.prediction.linear, testset$koi_disposition)
svm.table.polynomial = table(svm.prediction.polynomial, testset$koi_disposition)
svm.table.radial = table(svm.prediction.radial, testset$koi_disposition)
svm.table.sigmoid = table(svm.prediction.sigmoid, testset$koi_disposition)

# Risultati
print(paste("Accuracy SVM LINEAR: ", 
            sum(diag(svm.table.linear)) / sum(svm.table.linear)))
print(paste("Accuracy SVM POLYNOMIAL: ", 
            sum(diag(svm.table.polynomial)) / sum(svm.table.polynomial)))
print(paste("Accuracy SVM RADIAL: ", 
            sum(diag(svm.table.radial)) / sum(svm.table.radial)))
print(paste("Accuracy SVM SIGMOID: ", 
            sum(diag(svm.table.sigmoid)) / sum(svm.table.sigmoid)))

# CONFUSION MATRIX con statistiche
confusionMatrix(svm.prediction.linear, testset$koi_disposition,
                mode ="everything", positive = "CONFIRMED")
confusionMatrix(svm.prediction.polynomial, testset$koi_disposition,
                mode ="everything", positive = "CONFIRMED")
confusionMatrix(svm.prediction.radial, testset$koi_disposition,
                mode ="everything", positive = "CONFIRMED")
confusionMatrix(svm.prediction.sigmoid, testset$koi_disposition,
                mode ="everything", positive = "CONFIRMED")



#############################   Neural Network   ###############################


# Vengono specificati i nomi dei livelli per renderli validi per R
levels(trainset$koi_disposition) <- c("CONFIRMED", "FALSE_POSITIVE")
levels(testset$koi_disposition) <- c("CONFIRMED", "FALSE_POSITIVE")

# Training del modello
nn = train(koi_disposition ~ ., data = trainset, method = "nnet", metric = "ROC",
           trControl = trainControl(method = "cv", number = 1,
                                    classProbs = TRUE, 
                                    summaryFunction = twoClassSummary,
                                    verboseIter = TRUE))

# Predizione sul testset
neunet.probs = predict(nn, testset[,! names(testset) %in% c("koi_disposition")],
                       type = "prob")
neunet.prediction = ifelse(neunet.probs$CONFIRMED > neunet.probs$FALSE_POSITIVE,
                     "CONFIRMED", "FALSE_POSITIVE") 
neunet.prediction = factor(neunet.prediction)

# Risultati
neunet.table = table(neunet.prediction, testset$koi_disposition)
print(paste("Accuracy Neural Network: ", sum(diag(neunet.table)) / sum(neunet.table)))
# Confusion matrix con statistiche
confusionMatrix(neunet.prediction, testset$koi_disposition, mode ="everything",
                positive = "CONFIRMED")



##############################   Naive Bayes   #################################


# Training del modello
nb.model <- naiveBayes(koi_disposition ~ ., data = trainset, type = 'class')
# Predizione sul testset
nb.prediction <- predict(nb.model, testset, type = 'class')

# Risultati
nb.table = table(nb.prediction, testset$koi_disposition)
print(paste("Accuracy Naive Bayes: ", sum(diag(nb.table)) / sum(nb.table)))
# CONFUSION MATRIX con statistiche
confusionMatrix(nb.prediction, testset$koi_disposition, mode ="everything")



##############################   Decision Tree   ###############################


# Data Exploration
countTest = nrow(testset)
countTrain = nrow(trainset)
table(trainset$koi_disposition)
prop.table(table(trainset$koi_disposition))

### Creazione del decision tree senza limitazioni di crescita
decisionTree = rpart(koi_disposition ~ .,data = trainset, method = "class",
                     control = rpart.control(cp = 0))

# Visualizzazione dell'albero 1
plot(decisionTree)
# text(decisionTree)
plotcp(decisionTree)
fancyRpartPlot(decisionTree, sub = "")

# Test del modello 
testset$Prediction <- predict(decisionTree, testset, type = "class")

# Accuracy ~90%)
confusion.matrix = table(testset$Prediction, testset$koi_disposition)
confusion.matrix
sum(diag(confusion.matrix))/sum(confusion.matrix)

### Creazione del decision tree limitandone la profondità massima raggiungibile dai rami
decisionTree = rpart(koi_disposition ~ ., data = trainset, method = "class",
                     control = rpart.control(cp = 0, maxdepth = 5)) 

# Visualizzazione dell'albero 2
plot(decisionTree)
# text(decisionTree)
plotcp(decisionTree)
fancyRpartPlot(decisionTree, sub = "")

# Test del modello 
testset$Prediction <- predict(decisionTree, testset, type = "class")

# Accuracy (~91%)
confusion.matrix = table(testset$Prediction, testset$koi_disposition)
confusion.matrix
sum(diag(confusion.matrix))/sum(confusion.matrix)

### Provo a tagliare ulteriormente l'albero applicando un filtro al valore cp 
### (osservabile dal grafico generato da plotcp)
prunedDecisionTree = prune(decisionTree, cp = 0.013)

# Visualizzazione dell'albero 3
fancyRpartPlot(prunedDecisionTree, sub = "")

# Test del modello 
testset$Prediction <- predict(prunedDecisionTree, testset, type = "class")

# Accuracy (~ 90%)
confusion.matrix = table(testset$Prediction, testset$koi_disposition)
confusion.matrix
sum(diag(confusion.matrix))/sum(confusion.matrix)

# Creo una random forest per vedere se è possibile trovare un albero migliore
randomForest = randomForest(koi_disposition ~ ., data = trainset,
                            method = "class", ntree = 500)

# Test del modello 
testset$Prediction <- predict(randomForest, testset, type = "class")

# Accuracy (~ 93%)
confusion.matrix = table(testset$Prediction, testset$koi_disposition)
confusion.matrix
print(paste("Accuracy Random Forest: ",
            sum(diag(confusion.matrix))/sum(confusion.matrix)))

# Applicando la random forest l'accuracy migliora di circa il 2%



############################   Model Evaluation   ##############################


# 10-fold cross-validation con 3 ripetizioni
crossValidation = trainControl(method = "repeatedcv", number = 10, repeats = 3,
                       classProbs = TRUE, summaryFunction = twoClassSummary,
                       verboseIter = FALSE)

# Replacement dei valori del campo target
levels(trainset$koi_disposition) <- c("CONFIRMED", "FALSEPOSITIVE")
levels(testset$koi_disposition) <- c("CONFIRMED", "FALSEPOSITIVE")

# Train
svm.model= train(koi_disposition ~ ., data = trainset, method = "svmRadial",
                 metric = "ROC", trControl = crossValidation)
nn.model= train(koi_disposition ~ ., data = trainset, method = "nnet",
                metric = "ROC", trControl = crossValidation, trace = FALSE)
nb.model= train(koi_disposition ~ ., data = trainset, method = "naive_bayes",
                metric = "ROC", trControl = crossValidation)
rpart.model = train(koi_disposition ~ ., data = trainset, method = "rpart2",
                    metric = "ROC", trControl = crossValidation)
rf.model= train(koi_disposition ~ ., data = trainset, method = "rf",
                metric = "ROC", trControl = crossValidation)

# Prediction
svm.probs = predict(svm.model, testset[,! names(testset) %in% c("koi_disposition")],
                    type = "prob")
nn.probs = predict(nn.model, testset[,! names(testset) %in% c("koi_disposition")],
                   type = "prob")
nb.probs = predict(nb.model, testset[,! names(testset) %in% c("koi_disposition")],
                   type = "prob")
rpart.probs = predict(rpart.model, testset[,! names(testset) %in% c("koi_disposition")],
                      type = "prob")
rf.probs = predict(rf.model, testset[,! names(testset) %in% c("koi_disposition")],
                   type = "prob")


par(pty = "s") # Da eseguire prima di plottare le ROC

# ROC SVM
svm.ROC = roc(response = testset[,c("koi_disposition")],
              predictor = svm.probs$CONFIRMED,
              levels = levels(testset[,c("koi_disposition")]),
              plot = TRUE, legacy.axes = TRUE, percent = TRUE, 
              xlab = "% False Positive", ylab = "% True Positive",
              col = "red", lwd = 3)
title(main = paste("AUC SVM: ", svm.ROC$auc),
      line = -1.5, cex.main = 1, outer = TRUE)

# ROC Naive Bayes
nb.ROC = roc(response = testset[,c("koi_disposition")],
             predictor = nb.probs$CONFIRMED,
             levels = levels(testset[,c("koi_disposition")]),
             plot = TRUE, legacy.axes = TRUE, percent = TRUE, 
             xlab = "% False Positive", ylab = "% True Positive",
             col = "green", lwd = 3)
title(main = paste("AUC Naive Bayes: ", nb.ROC$auc),
      line = -1.5, cex.main = 1, outer = TRUE)

# ROC Neural Network
nn.ROC = roc(response = testset[,c("koi_disposition")],
             predictor = nn.probs$CONFIRMED,
             levels = levels(testset[,c("koi_disposition")]),
             plot = TRUE, legacy.axes = TRUE, percent = TRUE, 
             xlab = "% False Positive", ylab = "% True Positive",
             col = "blue", lwd = 3)
title(main = paste("AUC Neural Network: ", nn.ROC$auc),
      line = -1.5, cex.main = 1, outer = TRUE)

# ROC Decision Tree
rpart.ROC = roc(response = testset[,c("koi_disposition")],
                predictor = rpart.probs$CONFIRMED,
                levels = levels(testset[,c("koi_disposition")]),
                plot = TRUE, legacy.axes = TRUE, percent = TRUE, 
                xlab = "% False Positive", ylab = "% True Positive",
                col = "purple", lwd = 3)
title(main = paste("AUC Decision Tree: ", rpart.ROC$auc),
      line = -1.5, cex.main = 1, outer = TRUE)

# ROC Random Forest
rf.ROC = roc(response = testset[,c("koi_disposition")],
             predictor = rf.probs$CONFIRMED,
             levels = levels(testset[,c("koi_disposition")]),
             plot = TRUE, legacy.axes = TRUE, percent = TRUE, 
             xlab = "% False Positive", ylab = "% True Positive",
             col = "orange", lwd = 3)
title(main = paste("AUC Random Forest: ", rf.ROC$auc),
      line = -1.5, cex.main = 1, outer = TRUE)


# AUC
svm.ROC
nn.ROC
nb.ROC
rpart.ROC
rf.ROC
# Calcolo Performance
cv.values = resamples(list('SVM' = svm.model, 
                           'Naive Bayes' = nb.model, 
                           'Neural Network' = nn.model, 
                           'Decision Tree' = rpart.model, 
                           'Random Forest' = rf.model))
summary(cv.values)

# Plot vari sulle performace
dotplot(cv.values, metric = "ROC")
bwplot(cv.values, layout = c(3, 1))
splom(cv.values, metric = "ROC")

# Prime 15 variabili per importanza 
imp = varImp(svm.model)  # SVM
plot(imp, top = 15)
imp = varImp(nn.model)   # Neural Network
plot(imp, top = 15)
imp = varImp(nb.model)   # Naive Bayes
plot(imp, top = 15)
imp = varImp(rpar.model) # Decision Tree
plot(imp, top = 15)
imp = varImp(rf.model)   # Random Forest
plot(imp, top = 15)
# Timing
cv.values$timings


##################################   END   #####################################