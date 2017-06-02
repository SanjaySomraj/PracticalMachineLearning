# Prediction Analysis - Human Activity Recognition
Sanjay Somraj  
May 24, 2017  



## Executive Summary
Human Activity Recognition - HAR - has emerged as a key research area in the last years and is gaining increasing attention by the pervasive computing research community, especially for the development of context-aware systems. There are many potential applications for HAR, like: elderly monitoring, life log systems for monitoring energy expenditure and for supporting weight-loss programs, and digital assistants for weight lifting exercises

The approach proposed for the Weight Lifting Exercises dataset investigates "how (well)" an activity was performed by the wearer. The "how (well)" investigation has only received little attention so far, even though it potentially provides useful information for a large variety of applications,such as sports training.

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.  

Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience.

## Overview - Prediction analysis
This Prediction Analysis on Human Activity Recognition is divided into two main sections:

1)   Data analysis
2)   Build and Analyse prediction model

The **1st section** focuses on Data Analysis and pre-processing:

a)   Loading the requisite libraries and data
b)   Provide insights in to the data.
c)   Partition the training data.
d)   Cleaning data and Data transformations.

The **2nd section** focuses on training and building of prediction model.

a)   We will analyse different prediction models on training dataset **(myTraining)**.
b)   Conduct a dry-run for their accuracy on subset of training dataset **(myTesting)**.
c)   Cross validate the model in order to report an estimate of the out of sample error.
d)   Based on these, a model may be chosen to use to make predictions on the test datset **(testing)**.

## Data Analysis
### Loading the requisite libraries and data

```r
library(dplyr)
library(caret)
library(rpart)
library(randomForest)
library(knitr)
library(broom)

trainDataFileURL <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testDataFileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if (!file.exists("./data/pml-training.csv")) {
  download.file(trainDataFileURL, destfile="./data/pml-training.csv", method="curl")
}
if (!file.exists("./data/pml-testing.csv")) {
  download.file(testDataFileURL, destfile="./data/pml-testing.csv", method="curl")
}

trainData_As_is <- read.csv("./data/pml-training.csv", 
                            stringsAsFactors = FALSE, 
                            na.strings = c("NA","#DIV/0!",","))

testData_As_is <- read.csv("./data/pml-testing.csv", 
                           stringsAsFactors = FALSE, 
                           na.strings = c("NA","#DIV/0!",","))
```

### Exploratory Data Analysis
Let us look at what the Trainng and Testing data files have to say,


```r
dim(trainData_As_is)
dim(testData_As_is)

str(trainData_As_is)
str(testData_As_is)

problemID <- testData_As_is$problem_id
```
*    In the **training** data, there are **19622** observations on **160** categories. 
*    And the **testing** data has **20** observations on **160** categories.
*    The **testing** dataset does not have the **classe** variable present in the **training** data.
*    The **problem_id** from the **testing** data is a unique identfiier for each of the 20 observations. This column shall be bind with the **testing** dataset before the predictions on the **testing** dataset are done in the last section.

**NOTE:** There are many variables that are either NAs, DIV/0, or empty "".

### Cleaning Data and Data transformations
From the EDA on Training data, we understand there are a lot observations/rows which are not commplete and there are categories/columns which have no data but only missing values (**NAs**). These observations/rows are of no use and should be removed.  

**STEP 1:** Remove those columns which do not contribute to the predictions i.e. the columns that will not contribute to prediction. These are near zero covariates using the nearZeroVar function in **caret** package.


```r
nearZeroVars <- nearZeroVar(trainData_As_is, saveMetrics=TRUE)
trainData_As_is <- trainData_As_is[,nearZeroVars$nzv==FALSE]
```

We will create a new Training dataset (**myTraining**) and a new Test dataset (**myTesting**) by partitioning our original Training dataset (**trainingData_As_is**) in 60-40 ratio.


```r
inTrain <- createDataPartition(trainData_As_is$classe, p=0.6, list=FALSE)
myTraining <- trainData_As_is[inTrain, ]
myTesting <- trainData_As_is[-inTrain, ]
```

*    Our new Training dataset **myTraining** has **11776** rows and **124**
*    And our Testing dataset **myTesting** has **7846** rows and **124**

**STEP 2:** Remove the columns which have more than 70% missing values


```r
trainDataTemp <- myTraining
for(i in 1:length(myTraining)) {
     if( sum( is.na( myTraining[, i] ) ) /nrow(myTraining) >= .7) {
          for(j in 1:length(trainDataTemp)) {
               if( length( grep(names(myTraining[i]), names(trainDataTemp)[j]) ) == 1) {
                    trainDataTemp <- trainDataTemp[ , -j]
               }   
          } 
     }
}
myTraining <- trainDataTemp
rm(trainDataTemp)
```

**STEP 3:** We have to perform data transformations in **myTraining** and **myTesting** datasets- 
*    Remove the first column **X**
*    Convert **user_name** and **classe** variables from character into factor
*    Convert **cvtd_timestamp** from date/time to factor


```r
myTraining <- myTraining[c(-1)]
myTraining$user_name<-as.factor(myTraining$user_name)
myTraining$classe <- as.factor(myTraining$classe)
myTraining$cvtd_timestamp <- as.factor(weekdays(as.Date(myTraining$cvtd_timestamp)))
```

We also ensure that the **myTesting** data frame also has the same column structure .i.e.
*    **myTesting** has the same structure as ** myTraining**
*    we pick the same columns from orginal test dataframe **testData_As_is** into a new dataframe **testing**



We will check that the **testing** dataframe has same class, number of rows and columns are in the original **myTraining** dataframe, except for the **classe** column which was not present in the original **testing** dataframe.


```r
for (testCtr in 1:length(testing) ) {
     for(trainCtr in 1:length(myTraining)) {
          if( length( grep(names(myTraining[trainCtr]), names(testing)[testCtr]) ) == 1)  {
               class(testing[testCtr]) <- class(myTraining[trainCtr])
          }      
     }      
}

testing <- rbind(myTraining[2, -58], testing)
testing <- testing[-1,]
testing<-cbind(testing,problemID)
```

## Build and Analyse prediction models
### Prediction with RPart Decision trees

```r
set.seed(123)
modFit_rpart <- rpart(classe ~ ., data=myTraining, method="class")
predictions_rpart <- predict(modFit_rpart, myTesting, type="class")
confsmtrix_rpart <- confusionMatrix(predictions_rpart, myTesting$classe)
```
The overall prediction accuracy obtained through **RPART** is **83.291%**

### Prediction with Random Forest

```r
set.seed(123)
modFit_randFor <- randomForest(classe ~ ., data=myTraining)
predictions_randFor <- predict(modFit_randFor, myTesting, type="class")
confsmtrix_randFor <- confusionMatrix(predictions_randFor, myTesting$classe)
```
The overall prediction accuracy obtained through **Random Forest** is **99.834%**

### Prediction with GBM

```r
set.seed(123)
gbmTrainControl <- trainControl(method="repeatedcv", number=5, repeats=1)
modFit_GBM <- train(classe ~ ., data=myTraining, method ="gbm", 
                    trControl=gbmTrainControl, verbose=FALSE)
predictions_GBM <- predict(modFit_GBM, newdata=myTesting)
confsmtrix_GBM <- confusionMatrix(predictions_GBM, myTesting$classe)
```

The overall prediction accuracy obtained through **GBM** repeated cross-validation is **99.618%**

## Predictions on testing dataset
From the three prediction models we have tried, we can infer that the Random Forest had the highest accuracy level.


Model                              Overall Accuracy %
--------------------------------  -------------------
RF - Random Forest                             99.834
GBM - Gradient Boosting Machine                99.618
Rpart - Recursive Partitioning                 83.291

We will apply Random Forest model for prediction on the original **testing** dataset. The following are the results for the 20 observations in the **testing** dataset. The **Out-Of-Sample Error** expected would be **0.166%**
&nbsp;


```r
predictRandFor_testing <- predict(modFit_randFor, newdata=testing, type = "class")
testingPredictionResults <- data.frame(
     problem_id=testing$problemID,
     predicted=predictRandFor_testing
)
```


                  X1   X21   X3   X4   X5   X6   X7   X8   X9   X10   X11   X12   X13   X14   X15   X16   X17   X18   X19   X20 
----------------  ---  ----  ---  ---  ---  ---  ---  ---  ---  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----
Problem ID        1    2     3    4    5    6    7    8    9    10    11    12    13    14    15    16    17    18    19    20  
Predicted Value   B    A     B    A    A    E    D    B    A    A     B     C     B     A     E     E     A     B     B     B   

### Generating data file with prediction results

```r
filename = "./data/PredictedValues.csv"
write.csv(testingPredictionResults,file = filename, row.names = FALSE)
```

## Reference
The data for this analyis is taken from the WLE (Weight Lifting Exercise) dataset.  
**Read more:** http://groupware.les.inf.puc-rio.br/har  

**Thanks to:**  
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

