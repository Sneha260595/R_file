
install.packages('mice')
library(mice)
##input_Data <- read.csv("Downloads/fake_job_postings.csv", stringsAsFactors = FALSE)
input_Data<- read.csv("Downloads/fake_job_postings.csv", header=T, na.strings=c("","NA"))
colnames(input_Data)
head(input_Data)
colSums(is.na(input_Data))

library(dplyr)
library(tidyr)




### Training and test dataset creation:

NROW(input_Data)
colnames(input_Data)
#### Data cleaning:
#####]will delete the various columns:

##job_id because my DataFrame already has a built in index.
##salary_range because around 84% of the data is missing
##department because around 65% of the data is missing
##benefits because 40% of the data is missing
##company_profile because we can  combine the description + requirements columns to one features, in order to get only relevant data 
input_Data_clean<-input_Data[-c(1,2,3,5,6,9,4)] ## removed unnecessary columns :
colnames(input_Data_clean)






## Label encoding :


install.packages("ade4")
install.packages("dummies")
library(dummies)
columns_data= c('required_experience','required_education','industry', 'function')
Dataframe_new <- dummy.data.frame(input_Data_clean1, names = c("required_experience","required_education","industry","function.","employment_type") , sep = ".")
nrow(Dataframe_new)
colnames(Dataframe_new)
View(Dataframe_new)

Dataframe_new$Req_Descirption = paste(Dataframe_new$description,Dataframe_new$requirements)
Dataframe_new_Final<-Dataframe_new[-c(1,2)]
colnames(Dataframe_new_Final)
View(Dataframe_new_Final)
library(data.table)
library(mltools)



##DTM formation:
install.packages("corpus")
install.packages("tm")
require(tm)
library(corpus)
Dataframe_corpus <- VCorpus(VectorSource(Dataframe_new_Final$Req_Descirption))
DTM_Dataframe<- DocumentTermMatrix(Dataframe_corpus, control = list(
  tolower = TRUE,removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))

DTM_Dataframe
nrow(DTM_Dataframe)
X_dtm_train <- DTM_Dataframe[1:4169, ]
X_dtm_test  <- DTM_Dataframe[4170:7199, ]

X_train_labels <- Dataframe_new_Final[1:4169, ]$fraudulent
X_test_labels  <- Dataframe_new_Final[4170:7199, ]$fraudulent
prop.table(table(X_train_labels))
prop.table(table(X_test_labels))


## Naive Bayes:


findFreqTerms(X_dtm_train, 5)
freq_words <- findFreqTerms(X_dtm_train, 5)
str(freq_words)
freq_words_train<- X_dtm_train[ , freq_words]
freq_words_test <- X_dtm_test[ , freq_words]
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

X_train_final <- apply(freq_words_train, MARGIN = 2,
                   convert_counts)
X_test_final <- apply(freq_words_test, MARGIN = 2,
                    convert_counts)
install.packages("e1071") 
library(e1071)

## naive bayes
Job_classifier <- naiveBayes(X_train_final, X_train_labels)
Final_test_pred <- predict(Job_classifier, X_test_final)
library(gmodels)
CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))

### Mice package
#colSums(is.na(input_Data_clean))
#sample1 = input_Data_clean[,c(2,4,8,9,10,11,12)]
#input_Data_clean1 = mice(sample1, m=1)


input_Data_clean1<-na.omit(input_Data_clean)
colSums(is.na(input_Data_clean1))
View(input_Data_clean1)
# Random sample indexes
train_index <- sample(1:nrow(input_Data_clean1), 0.8 * nrow(input_Data_clean1))
test_index <- setdiff(1:nrow(input_Data_clean1), train_index)
colSums(is.na(X_test))
# Build test labels and train splits
X_train <- input_Data_clean1[train_index, -13]
X_train_label <- input_Data_clean1[train_index, "fraudulent"]

X_test <- input_Data_clean1[test_index, -13]
X_test_label <- input_Data_clean1[test_index, "fraudulent"]

## KNN algorithm
install.packages("class")
library(class)
Input_data_Pred <- knn(train = X_train, test = X_test,
                      cl = X_train_label, k = 10)

### Random Forest :
library(randomForest)
set.seed(300)
rf <- randomForest(fraudulent ~ ., data = input_Data_clean1)
