## load data
library(AppliedPredictiveModeling)
library(caret)
data(AlzheimerDisease)

## overview data
dim(predictors) # 333 130
head(predictors[,1:5])
summary(diagnosis)

## pre-process genotype
predictors$Genotype
summary(predictors$Genotype)
predictors$E2 = predictors$E3 = predictors$E4 = 0
predictors$E2[grepl("2", predictors$Genotype)] = 1
predictors$E3[grepl("3", predictors$Genotype)] = 1
predictors$E4[grepl("4", predictors$Genotype)] = 1

## split the data using stratified sampling
split = createDataPartition(diagnosis, p = 0.8, list = FALSE) # in caret package
## combine into one data frame
adData = predictors
adData$Class = diagnosis
training = adData[split, ]
testing = adData[-split, ]
## save a vector of predictor variable names
predVars = names(adData)[!(names(adData) %in% c("Class", "Genotype"))]

## compute the area under ROC curve, sensitivity, specificity, accuracy and Kappa
fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))

## create resampling data sets to use for all models
set.seed(104)
index = createMultiFolds(training$Class, times = 5)

## create a vector of subset sizes to evaluate
varSeq <- seq(1, length(predVars)-1, by = 2)

## ######################################################
## forward, backward, and stepwise selection
## ######################################################

initial <- glm(Class ~ tau + VEGF + E4 + IL_3, data = training, family = binomial)
library(MASS)
stepAIC(initial, direction = "both")

## ######################################################
## recursive feature elimination
## ######################################################

str(rfFuncs)
newRF = rfFuncs
newRF$summary = fiveStats
## control function
ctrl <- rfeControl(method = "repeatedcv",
                   repeats = 5,
                   verbose = TRUE,
                   functions = newRF,
                   index = index)
set.seed(721)
rfRFE <- rfe(x = training[, predVars],
             y = training$Class,
             sizes = varSeq,
             metric = "ROC",
             rfeControl = ctrl,
             ## pass options to randomForest()
             ntree = 1000)
rfRFE

## predict
predict(rfRFE, head(testing))

## fit svm
svmFuncs = caretFuncs
svmFuncs$summary = fiveStats
ctrl <- rfeControl(method = "repeatedcv",
                   repeats = 5,
                   verbose = TRUE,
                   functions = svmFuncs,
                   index = index)
set.seed(721)
svmRFE <- rfe(x = training[, predVars],
              y = training$Class,
              sizes = varSeq,
              metric = "ROC",
              rfeControl = ctrl,
              ## now options to train()
              method = "svmRadial",
              tuneLength = 12,
              preProc = c("center", "scale"),
              ## below specifies the inner resampling process
              trControl = trainControl(method = "cv",
                                       verboseIter = FALSE,
                                       classProbs = TRUE))

## ######################################################
## filter methods
## ######################################################

## to compute a p-value for each predictor
pScore <- function(x, y)
{
  numX = length(unique(x))
  if (numX > 2)
  {
    ## with many values in x, compute a t-test
    out = t.test(x ~ y)$p.value
  }
  else
  {
    ## for binary predictors, test the odds ratio == 1 via
    ## fisher's exact test
    out = fisher.test(factor(x), y)$p.value
  }
  out
}
## apply the scores to each of the predictors columns
scores <- apply(X = training[, predVars],
                MARGIN = 2,
                FUN = pScore,
                y = training$Class)
tail(scores)