library(dplyr)
library(ggplot2)
library(caret)
library(MASS)
library(rpart)
library(randomForest)

setwd("D:/Sanjay S/KM/DataScienceProjects/Practical Machine Learning")

trainData_As_is <- read.csv("./data/pml-training.csv", stringsAsFactors = FALSE, na.strings = c("NA","#DIV/0!",","))
testData_As_is <- read.csv("./data/pml-testing.csv", stringsAsFactors = FALSE,  na.strings = c("NA","#DIV/0!",","))

problemID <- testData_As_is$problem_id

nearZeroVars <- nearZeroVar(trainData_As_is, saveMetrics=TRUE)
trainData_As_is <- trainData_As_is[,nearZeroVars$nzv==FALSE]

dim(trainData_As_is)

inTrain <- createDataPartition(trainData_As_is$classe, p=0.6, list=FALSE)
myTraining <- trainData_As_is[inTrain, ]
myTesting <- trainData_As_is[-inTrain, ]

#nearZeroVars<- nearZeroVar(myTesting,saveMetrics=TRUE)
#myTesting <- myTesting[,nearZeroVars$nzv==FALSE]

dim(myTraining) 
dim(myTesting)

trainDataTemp <- myTraining
for(i in 1:length(myTraining)) {
     if( sum( is.na( myTraining[, i] ) ) /nrow(myTraining) >= .7) {
          for(j in 1:length(trainDataTemp)) {
               if( length( grep(names(myTraining[i]), names(trainDataTemp)[j]) ) == 1)  {
                    trainDataTemp <- trainDataTemp[ , -j]
               }   
          } 
     }
}

myTraining <- trainDataTemp
rm(trainDataTemp)

myTraining <- myTraining[c(-1)]
myTraining$user_name<-as.factor(myTraining$user_name)
myTraining$cvtd_timestamp <- as.factor(weekdays(as.Date(myTraining$cvtd_timestamp)))
myTraining$classe <- as.factor(myTraining$classe)

myTesting$user_name<-as.factor(myTesting$user_name)
myTesting$cvtd_timestamp <- as.factor(weekdays(as.Date(myTesting$cvtd_timestamp)))
myTesting$classe <- as.factor(myTesting$classe)

dim(myTraining)

myTestCols <- colnames(myTraining)
testCols <- colnames(myTraining[, -58])

myTesting <- myTesting[myTestCols]
testing <- testData_As_is[testCols]

dim(myTraining)
dim(myTesting)
dim(testing)

testing$cvtd_timestamp <- as.factor(weekdays(as.Date(testing$cvtd_timestamp)))

for (testCtr in 1:length(testing) ) {
     for(trainCtr in 1:length(myTraining)) {
          if( length( grep(names(myTraining[trainCtr]), names(testing)[testCtr]) ) == 1)  {
                    class(testing[testCtr]) <- class(myTraining[trainCtr])
          }      
     }      
}

testing <- rbind(myTraining[2, -58], testing)
testing <- testing[-1,]
testingx <- testing
testing<-cbind(testing,problemID)

set.seed(123)
modFit_rpart <- rpart(classe ~ ., data=myTraining, method="class")
predictions_rpart <- predict(modFit_rpart, myTesting, type="class")
confsmtrix_rpart <- confusionMatrix(predictions_rpart, myTesting$classe)
confsmtrix_rpart

set.seed(123)
modFit_randFor <- randomForest(classe ~ ., data=myTraining)
predictions_randFor <- predict(modFit_randFor, myTesting, type="class")
confsmtrix_randFor <- confusionMatrix(predictions_randFor, myTesting$classe)
confsmtrix_randFor

set.seed(123)
fitControl <- trainControl(method="repeatedcv", number=5, repeats=1)
modFit_GBM <- train(classe ~ ., data=myTraining, method ="gbm", trControl=fitControl, verbose=FALSE)
predictions_GBM <- predict(modFit_GBM, newdata=myTesting)
confsmtrix_GBM <- confusionMatrix(predictions_GBM, myTesting$classe)
confsmtrix_GBM

predictRandFor_testing <- predict(modFit_randFor, newdata=testing, type = "class")
testingPredictionResults <- data.frame(
     problem_id=testing$problemID,
     predicted=predictRandFor_testing
)
colnames(testingPredictionResults)<-c("Problem ID","Predicted Value")

print(t(testingPredictionResults), col.names = FALSE)

filename2 = "Problem_ID.csv"
write.csv(testingPredictionResults,file = filename2, row.names = FALSE)
