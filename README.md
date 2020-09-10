# R_file
Staffing firm (classification of jobs)















































Experiential Project 

 



































Authored by: Sneha
 

Prediction of Fake/Genuine job postings
This data is downloaded from below link. It contains information about job postings. https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction/kernels
We will be developing machine learning models to classify the job listings as Fraudulent and Non-Fraudulent.

Data Load and Exploratory Data Analysis and Data Preparation

Below is the code to load data:

Once the data is loaded, let us view the data structure and dimensions
 


 



Summary of Data
 


 

There are 17880 observations and 18 features in the dataset. All these 18 features contain various aspects of a job posting and if the posting is fake or not. The different features include job title, job location, job department, fraudulent salary range, employment type, required experience, required education, industry, function, and several others. Column “Fraudulent” tells if the posted job is fake or not. 0 represents the job is genuine and 1 represents a fake job posting. For our Models, Fraudulent feature is our response variable and the remaining 17 features are our independent features. Let us look at the number of Fraudulent and number of Genuine job postings:

 

We can see that there are 17014 genuine job postings and 866 fraud postings. The data is bias towards genuine postings.

Further investigating the data, lets look at the jobs per required education colored by required experience. We see that the highest number of job postings did not have required education and experience listed in the listing. Most of the entry level jobs required high school equivalent which makes sense.

 



Feature Extraction from Job Description column

We need to extract the features from Job description column to identify the words that are specific and most frequent in each of Fraudulent and non-fraudulent job postings.
Text mining library tm can be used to modify the texts and extract individual words. Below is the code:

We need to convert the “Description” column text to lower font and then remove special characters, numbers. White spaces and perform stemming.

Below is the result:



Using wordcloud(), we can form a word cloud with the most frequent words highlighted with larger font. Below is the code for Non-Fraudulent word cloud.
 




Code:


Result:



Similarly, now we will find the most frequent words in Fraudulent job postings:
 


 

Result:

Below is the code to view data in word cloud format:

Result:
 



 

Data Cleaning and One hot Encoding for categorical variables:

This is the most important part that needs to be done before modelling. As this data has a lot of unimportant information. I have initially checked what are all the relevant details for further modelling and found certain columns can be eliminated as they do not add any value on the importance of classification of a job listing into genuine and fraudulent categories. I have found that
1.	Job_id is not necessary because our DataFrame already has a built in index.
2.	Salary_range because around 84% of the data is missing.
3.	Department because around 75% of the data is missing.
4.	Benefits because 40% of the data is missing and also it doesn't help us to find the relation of this independent variable with our dependent variable fraudulent.
5.	We have also removed the column company_profile as the relevant columns “description” and “requirements” which when clubbed form the basis for our classification.

Hence, We have initially removed the above listed columns from our dataframe and then worked on removal of NAN values using the “MICE” package as the categorical variables are binary in nature.
Hence, We have made sure to remove the NAN values and chose a smaller set of data for our further analysis as the modelling takes lot of computational time for large dataset with such huge character type data.And later we have create created a dummy frame for encoding our categorical variables as our model cannot accept the data in categorical form directly. There has to be a means that it gets
 

converted into integer values as most machine learning algorithms require numerical input and output variables. That an integer and
one hot encoding is used to convert categorical data to integer data.

One hot encoding has given us a resultant matrix wherein all categorical data have been replaced by encoded data. Further part of our analysis went ahead with lemmatization of the words from a combined column of “description” and “requirements” column. This way we can get to know the frequency of words and can be linked to classification.

Following is the code that explains data cleaning, One hot encoding and lemmatization(words frequency calculation) which was the most important and difficult task for us:


 




Result:

Before data cleaning:

After Data cleaning:


After one hot encoding: (all the columns were encoded except description and requirements which were combined to form another column called req_description, which are further analyzed for frequency of words calculation to create a uniform matrix that can be modelled in the next step.


 


 

Model Building

We will be working on three models: KNN, Naive Bayes and Random Forest on our dataset to classify the job postings as Fraudulent and Non-Fraudulent. Column “Fraudulent” is our dependent variable and the remaining 18 variables are our independent variables.

KNN Model - K-nearest neighbor: K-nearest neighbor algorithm is a supervised learning algorithm that can be used for classification and regression models.

Naïve Bayes: The Naive Bayes algorithm describes a simple method to apply Bayes' theorem to classification problems. It is a classification algorithm for binary and multi-class classification problems. Naive Bayes assumes that all of the features in the dataset are equally important and independent.

Random Forest: Random forest is the combination of many decision trees into a single model. Individually, predictions made by decision trees may not be accurate, but combined together, the predictions will be closer to the mark on average. “Each decision tree in the forest considers a random subset of features when forming questions and only has access to a random set of the training data points. When it comes time to make a prediction, the random forest takes an average of all the individual decision tree estimates” (Koehrsen W., 2017). It is not only used as a classification model but also performs regression by finding the most optimized value by taking the average of all the individual decision tree estimates.

Creating Training & Testing Dataset and Training the Model Algorithm1: Naïve Bayes:
 

Since our dataset has some of the columns that had many missing values, we have compiled them and took a subset of data for further analysis as nearly 18,000 data can take longer time to complete the model evaluation, columns from the data. After creation of frequency of words upon lemmatization. We have created a Document Term Matrix A document-term matrix or term-document matrix is a mathematical matrix that describes the frequency of terms that occur in a collection of documents. In a document-term matrix, rows correspond to documents in the collection and columns correspond to terms.We were good to proceed further with our analysis on modelling data using Naive Bayes classifier as follows: (Also to be noted we have considered a sample size of 8000 for our analysis)



 


 


Below is the code to apply Naïve Bayes Algorithm. We will be installing and loading the e1071 package.

Once we train our model on Train and Test dataset on Naïve Bayes algorithm, we can verify using CrossTable() function.



Result:

The following are the results from Cross table and it is quite evident that the number of False positives and false negatives are high and the accuracy of the model is also pretty low.From this confusion matrix we can get info regarding certain measures that help us to validate the model:

●	Precision: ->91.45 %
●	Recall:-->92%
●	F1-score-->90.89%
●	Accuracy-->84%
 


 






Algorithm2 : KNN classification:

On the same dataset we have used the same training and test sample to classify the data using KNN classification and have attained the following output. We have acuired the output for n value equal to 10-fold.
Since using Knn in R is a straightforward method to evaluate the model , following is the code representing the Knn classifier:


Result is as follows:

The following are the results from Cross table and it is quite evident that the number of False positives are 10 which is a pretty low number but false negatives being high about 70.

From this confusion matrix we can get info regarding certain measures that help us to validate the model:
 


●	Precision:(True positive/Total Predicted Positive) -> 95%
●	recall:(true Positive/Total Actual Positive)-->96%
●	F1-score(2*Precision*recall/(Precision+Recall)-->94.75%
●	Accuracy(Tp+Tn/Overall)-->94%


Algorithm3 : Random Forest classification:

Below is the code to apply the Random Forest Algorithm. We will be installing and loading the random forest package. And we have used the same set of data as shown previously( same train and test dataset). Following is the code representing modelling of data:



Result:


The following are the results from Cross table and it is quite evident that the number of False positives are 10 which is pretty low number but false negatives being high about 70. This matrix reflects the out-of-bag error rate (listed in the output as OOB estimate of error rate), which unlike resubstitution error, is an unbiased estimate of the test set error. This means that it should be a fairly reasonable estimate of future performance.





From this confusion matrix we can get info regarding certain measures that help us to validate the model:

●	Precision:(True positive/Total Predicted Positive) -> 98.6%
●	recall:(true Positive/Total Actual Positive)-->99%
 

●	F1-score(2*Precision*recall/(Precision+Recall)-->98%
●	Accuracy(Tp+Tn/Overall)-->97%










CONCLUSION:

We have performed the data cleaning, EDA for inferences, One hot encoding for encoding the categorical variable, lemmatized the words and finally evaluated our model using three classification models such as Naive Bayes, Knn and Random Forest algorithm and we have found that Random Forest algorithm is the most compliant and trusted model explaining greater variance of the data with almost 97% accuracy. And Naive Bayes algorithm has least accuracy out of three algorithms used.

Below is the table differentiating three algorithms:






References:
 

Predictive Analytics Using R by Jeffrey Strickland (2014). Retrieved
from https://www.slideshare.net/JeffreyStricklandPhD/predictiveanalyticsusingrredc
