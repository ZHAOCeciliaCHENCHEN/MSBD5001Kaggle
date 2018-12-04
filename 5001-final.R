# install.packages("caTools")
# install.packages("glmnet")
# install.packages("caret")
# install.packages("gbm")
library(caTools)
library(glmnet)
library(caret)
library(gbm)
library(readr)

# Import the data 
train0 <- read_csv("~/Desktop/all/train.csv")
test0 <- read_csv("/Users/zhaochenchen/Desktop/all/test.csv")
# Remove the feature id
train <- train0[, -1]
test <- test0[, -1]
train <- train[, -5]
test <- test[, -5]
# Change penalty into dummy variables
train$penalty <- as.factor(train$penalty)
dummy.penalty <- model.matrix(~ penalty - 1, data = train)
head(dummy.penalty)
train1 <- cbind(train[, -1], dummy.penalty)
train1$n_jobs[train1$n_jobs == '-1'] = 16
new1 <- (train1$n_samples*train1$n_features)/train1$n_jobs
new2 <- (train1$n_samples*train1$n_features)
new3 <- (train1$n_classes*train1$n_clusters_per_class)
train1$alpha <- -log10(train1$alpha)
train1 <- cbind(train1, new1, new2, new3)

nameList <- colnames(train1)
categoryList <- c("penalty")
train.target <- train$time


# XGBoost 
library(xgboost)
set.seed(1234)
xgbFit <- xgboost(data = as.matrix(train1[1:380, -12]), nfold = 10, 
                  label = as.matrix(train.target[1:380]), nrounds = 3000, verbose = 0,
                  objective = 'reg:gamma', val_metric = 'rmse')
# Predictions
preds.xgb <- predict(xgbFit, newdata = as.matrix(subset(train1, select = -time)[381:400, ]))
mean((train1[381:400, 12]- preds.xgb)^2)
importance.xgb <- xgb.importance(feature_names = colnames(subset(train1, select = -time)), model = xgbFit)
xgb.plot.importance(importance_matrix = importance.xgb)

feature.xgb <- importance.xgb[importance.xgb$Gain >= 0.003, Feature]
feature.xgb
set.seed(1234)
xgbFit1 <- xgboost(data = as.matrix(train1[feature.xgb][1:380, ]), nfold = 10, 
                   label = as.matrix(train.target[1:380]), nrounds = 3000, verbose = 0,
                   objective = 'reg:gamma', val_metric = 'rmse', eta = 0.09273402,
                   gamma = 0.3628627, max_depth = 6, min_child_weight = 1, subsample = 0.7818276, 
                   colsample_bytree = 0.6221984, max_delta_step = 9)
# Predictions
preds.xgb1 <- predict(xgbFit1, newdata = as.matrix(train1[feature.xgb][381:400, ]))
mean((subset(train1, select = time)[381:400, ]-preds.xgb1)^2)
dtrain <- xgb.DMatrix(as.matrix(train1[feature.xgb][1:380, ]), label = as.matrix(train.target[1:380]))
best_param = list()
best_seednumber = 1234
best_rmse = Inf
best_rmse_index = 0
for (iter in 1:1000) {
  param <- list(objective = "reg:gamma",
                eval_metric = "rmse",
                # num_class = 12,
                max_depth = sample(3:10, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.4), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
  )
  cv.nround <- 1000
  cv.nfold <- 5
  seed.number <- sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data=dtrain, params = param, nthread=6, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = T, early.stop.round=8, maximize=FALSE)
  
  min_rmse <- min(mdcv$evaluation_log$test_rmse_mean)
  min_rmse_index <- which.min(mdcv$evaluation_log$test_rmse_mean)
  
  if (min_rmse < best_rmse) {
    best_rmse = min_rmse
    best_rmse_index = min_rmse_index
    best_seednumber = seed.number
    best_param = param
  }
}
xgbFit2 <- xgboost(data = as.matrix(train1[feature.xgb]), nfold = 10, 
                   label = as.matrix(train.target), nrounds = 3000, verbose = 0,
                   objective = 'reg:gamma', val_metric = 'rmse', eta = 0.07784344,
                   gamma = 0.1236395, max_depth = 8, min_child_weight = 36, subsample = 0.7373081, 
                   colsample_bytree = 0.586284, max_delta_step = 7)
# nround = best_rmse_index
# set.seed(best_seednumber)
# xgb.model1 <- xgb.train(data = dtrain, params = best_param, nrounds = nround, nthread = 6)
# preds.xgb.model1 <- predict(xgb.model1, newdata = as.matrix(train1[feature.xgb][381:400, ]))
# mean((subset(train1, select = time)[381:400, ]-preds.xgb.model1)^2)

# Lasso
x <- model.matrix(time~.,train1)
y <- train1$time
lasso.mod <- glmnet(x[1:380, ], y[1:380], alpha = 1)
plot(lasso.mod)
set.seed(1234)
cv.out <- cv.glmnet(x[1:380, ], y[1:380], alpha = 1, type.measure = 'mse', nfolds = 10)
plot(cv.out)
bestlam <- cv.out$lambda.min
lasso.pred <-predict(lasso.mod, s = bestlam, newx = x[381:400,], alpha = 1)
mean((lasso.pred-y[381:400])^2)

# Weight
w <- seq(from = 0, to = 1, 0.00001)
best_rmse = Inf
best_rmse_index = 0
# After get the parameters from cv, it is required to change the value for both xgbFit1 and xgbFit2
# Run xgbFit1 and preds.xgb1 again, we use preds.xgb1 to decide the weight as following
for (i in 1:length(w)){
  pred <- w[i] * preds.xgb1 + (1 - w[i]) * lasso.pred
  rmse <- mean((subset(train1, select = time)[381:400, ] - pred) ^ 2)
  if (rmse < best_rmse){
    best_rmse <- rmse
    best_rmse_index <- i
  }
}
w_best <- c(w[best_rmse_index], 1 - w[best_rmse_index])
best_rmse
w_best

# Test
dummies <- dummyVars(~ penalty, data = test0)
pred.dummies <- as.data.frame(predict(dummies, newdata = test0))
test1 <- cbind(test[, -1], pred.dummies)
test1$n_jobs[test1$n_jobs == '-1'] = 16
test1$alpha <- -log10(test1$alpha)
new1 <- (test1$n_samples*test1$n_features)/test1$n_jobs
new2 <- (test1$n_samples*test1$n_features)
new3 <- (test1$n_classes*test1$n_clusters_per_class)
test1 <- cbind(test1, new1, new2, new3)
xt <- model.matrix(~.,test1)
preds.xgb2 <- predict(xgbFit2, newdata = as.matrix(test1[feature.xgb]))
lasso.pred1 <- predict(lasso.mod, s = bestlam, newx = xt, alpha = 1)

# Final Result
# Based on 'w_best'
result <- 0.9226*preds.xgb2 + 0.0774*lasso.pred1 
# If the result less than zero, use value got from XGBoost
for (i in 1:100){
  if (result[i] > 0){
    result[i] = result[i]}
  else{
    result[i] = preds.xgb2[i]}
}
write.table(result, file = "/Users/zhaochenchen/Desktop/zcc.csv", row.names = T, col.names = T, sep = ",")
