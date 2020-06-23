#importing libraries
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score,accuracy_score,classification_report
from sklearn.naive_bayes import MultinomialNB
import nltk
import re
import pickle

#reading csv
df = pd.read_table("Hotel.csv",delimiter=",")
print("Data has been read successfully....")
#converting the the rating above 4 and above 4 to 1 and below 4 to 0 
def filter_data(value):
    if value>=4.0:
        return 1
    else:
        return 0
#droping all the missing value columns because we have enough data
df = df.dropna()
print("the rows the contained missing values has been dropped successfully....")
"""
in the statement ==> df.reset_index(drop=True,inplace=True)
what we actually do is we reset the index number in the dataset because 
if we don't do it the column numbers will be disordered.
if we don't do we will have index number like 0,1,2,3,7,10,11,17.
if we have like this iteration is not possible we will get key value error
"""
df.reset_index(drop=True,inplace=True)
print("the index has been reseted....")
#creating a new column in our dataset which will have 0 and 1
df['modified'] = df['reviews.rating'].apply(filter_data)
nltk.download('stopwords')
valid = []
#the value is 35028 because we have that many rows after dropping
print("The looping has started....")
for i in range(0, 35028):
    review = re.sub('[^a-zA-Z]', ' ', df['reviews.text'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    review = ' '.join(review)
    valid.append(review)  
#vectorizing the features
print("The looping has ended....")
print("vectorization has started....")
vectorize = CountVectorizer(max_features = 1500)
#applying the vectorization and separating features and labels
features = vectorize.fit_transform(valid).toarray()
labels = df.iloc[:,-1].values
print("ended....")
#separating the train,test dataset
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 41)
#multinomial
classi = MultinomialNB()
print("Bernoulli naive bayes fitting has started....")
classi.fit(features_train, labels_train)
print("fitting has been done and prediction has stated")
labels_pred = classi.predict(features_test)
print("prediction has been done calculating confusion matrix")
#confusion matrix
naive_multinomial_confusion = confusion_matrix(labels_test, labels_pred)
print("-------------------------------------")
print("Multinomial naive bayes\n")
print("Confusion matrix\n",naive_multinomial_confusion)

print("Multinomial")
print("Accuracy",accuracy_score(labels_test,labels_pred))
print("precision score when 0 is considered",precision_score(labels_test, labels_pred, pos_label=0))
print("precision score when 1 is considered",precision_score(labels_test, labels_pred, pos_label=1))
print("recall score when 0 is considered",recall_score(labels_test, labels_pred, pos_label=0))
print("recall score when 1 is considered",recall_score(labels_test, labels_pred, pos_label=1))
print("f1 score when 0 is considered",f1_score(labels_test, labels_pred, pos_label=0))
print("f1 score when 1 is considered",f1_score(labels_test, labels_pred, pos_label=1))

"""
Results of various models

-------------------------------------
Bernoulli Naive bayes

Confusion matrix
 [[1519  900]
 [ 554 4033]]
Bernoulli naive bayes fitting has started....
fitting has been done and prediction has stated
prediction has been done calculating confusion matrix
-------------------------------------
Multinomial naive bayes

Confusion matrix
 [[1664  755]
 [ 580 4007]]
Random forest fitting has started....
fitting has been done and prediction has stated
prediction has been done calculating confusion matrix
-------------------------------------
Random forest classifier

Confusion matrix
 [[1538  881]
 [ 459 4128]]
Binomial
Accuracy 0.7924636026263203
precision score when 0 is considered 0.7327544621321755
precision score when 1 is considered 0.8175552402189337
recall score when 0 is considered 0.6279454319966928
recall score when 1 is considered 0.8792238936123828
f1 score when 0 is considered 0.676313446126447
f1 score when 1 is considered 0.8472689075630251
Multinomial
Accuracy 0.8094490436768484
precision score when 0 is considered 0.7415329768270945
precision score when 1 is considered 0.8414531709365812
recall score when 0 is considered 0.6878875568416701
recall score when 1 is considered 0.8735557008938304
f1 score when 0 is considered 0.713703624276217
f1 score when 1 is considered 0.857203979035191
Random Forest
Accuracy 0.8087353696831288
precision score when 0 is considered 0.7701552328492739
precision score when 1 is considered 0.8241165901377521
recall score when 0 is considered 0.6357999173212071
recall score when 1 is considered 0.8999345977763243
f1 score when 0 is considered 0.6965579710144927
f1 score when 1 is considered 0.8603584827011254


finally for this dataset the model we choose Multinomial naive bayes
because of it's merely high accuaracy.'
"""
"""
multinomial naive bayes Results:
    
    Multinomial naive bayes

Confusion matrix
 [[1664  755]
 [ 580 4007]]
 
 
 Multinomial
Accuracy 0.8094490436768484
precision score when 0 is considered 0.7415329768270945
precision score when 1 is considered 0.8414531709365812
recall score when 0 is considered 0.6878875568416701
recall score when 1 is considered 0.8735557008938304
f1 score when 0 is considered 0.713703624276217
f1 score when 1 is considered 0.857203979035191
"""
# Applying k-Fold Cross Validation
print("---------k-Fold Cross Validation-------------")
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classi, X = features_train, y = labels_train, cv = 20)
print ("accuracies is ", accuracies)
print ("mean accuracy is",accuracies.mean())
print ("std  accuracy is",accuracies.std())
print(classification_report(labels_test, labels_pred))

"""
k-fold cross validation report:
accuracies is  [0.81027104 0.82097004 0.8137045  0.80085653 0.80585296 0.8137045
 0.82155603 0.81441827 0.8137045  0.79728765 0.81084939 0.80513919
 0.81798715 0.79800143 0.82655246 0.81441827 0.78943612 0.81870093
 0.80442541 0.81727338]
mean accuracy is 0.8107554874702296
std  accuracy is 0.009148250247233003

classification report:
              precision    recall  f1-score   support

           0       0.74      0.69      0.71      2419
           1       0.84      0.87      0.86      4587

    accuracy                           0.81      7006
   macro avg       0.79      0.78      0.79      7006
weighted avg       0.81      0.81      0.81      7006

"""

data = []
def user_data(value):
    value1 = re.sub('[^a-zA-Z]', ' ', value)
    value1 = value1.lower()
    value1 = value1.split()
    value1 = [word for word in value1 if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    value1 = [ps.stem(word) for word in value1]
    value1 = ' '.join(value1)
    print(value1)
    data.append(value1)





user_data("nice hotel expensive parking got good deal stay hotel anniversary arrived late evening took advice previous reviews did valet parking check quick easy little disappointed non-existent view room room clean nice size bed comfortable woke stiff neck high pillows not soundproof like heard music room night morning loud bangs doors opening closing hear people talking hallway maybe just noisy neighbors aveda bath products nice did not goldfish stay nice touch taken advantage staying longer location great walking distance shopping overall nice experience having pay 40 parking night")

features = vectorize.fit_transform(data).toarray()
pred = classi.predict(features)


"""
Created on Tue Jun 16 14:43:15 2020

@author: sesha
"""


