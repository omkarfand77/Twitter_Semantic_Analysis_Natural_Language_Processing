# **Twitter_Semantic_Analysis_Natural_Language_Processing**

The Data Contains following Coloumns:
Tweet: The text of the tweet
Class: The respective class to which the tweet belongs. There are 4 classes -:

1. Regular
2. Sarcasm
3. Figurative (both irony and sarcasm)
4. Irony
*Here dependent is Class of tweet and Independent is Tweet*

*Business Objective:*

1. Need to get sentiment analysis of tweets gauge its impact and type Architecture level analysis.
2. Data transformation/Text processing using R/Python
3. Need to get sentiments Analysis and Emotion mining with some charts like histogram, Density plot, Barplot, pie-plot etc.
4. Deployment through Flask/ Streamlit

## Preprocessing

1. Removing URL from text data
2. Removing Special Characters
3. Converting text data into lower case
4. Removing stopwords
5. tokenization
6. Lemmatization

## **Exloratory Data Analysis (EDA)**

By applying `Bag of Words`, TF-IDF and unigram, Bigram, Trigram Extracted max 200 feature showed graphical representation with Wordcloud, histogram and barplot.

## **Word Embedding**
**`Word2Vec`**: The conversion of tweets text data is performed to understand the context of words
in text data from which the relevant class of data can be detrmined. 
For the Vectorization of words the glove's pretrained model is used.

**Feature Engineering and Feature Selection**
After the Vectorization of text data further it is classified into dependant and independant form.
Here dependant is class of data and independant is tweet. The MinMax Scaler is applied to normalize
the training data of a Machine Learning model, i.e. to bring the numerical values to a uniform scale.
next the dataset is splitted into the train_test split for evaluation of model performance in 80:20 ratio.


## **Model Building**
**Machine Learning**
Classification Model (Multi-class Classifier):
1. SVM
2. K-Nearest Neighbours
3. Kernel SVM
4. Naïve Bayes
5. Decision Tree Classification
6. Random Forest Classification

**Deep Learning** 
Artificial Neural Network: For bettwe performance of model accuracy and accurate prediction kearas sequential model building performed.

## **Model Evaluation**

1. Log Loss or Cross-Entropy Loss
2. Confusion Matrix:
3. AUC-ROC curve:

## **Model Selection**
On basis of accuracy score and relavent model predicton the kearas sequential model is used for deployment.
For deployment we made model pickle file as model.pkl. 

## **Deployment**
The Deployment is made throught flask on local host. The file app.py contain required for code of requesting text data and predicting
relevant class of tweet.
