####Amitabh Chakravorty
####Project: SVM Polynomial

import pandas as pd
import numpy as np
import statistics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import metrics

#Reading the training data
df = pd.read_csv('project_movie_data.csv')

df1 = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

#Determining the 6 edges or cut-off points of 5 bins/groups for the continuous data
lowest = df1['gross'].min() #Minimum gross revenue from the data, the 1st edge/cut-off point
average = df1['gross'].mean() #Average gross revenue from the data
second = (average - lowest)/3 #2nd edge/cut-off point
third = 2*((average - lowest)/3) #3rd edge/cut-off point
highest = df1['gross'].max() #Maximum gross revenue from the data, the last or 6th edge/cut-off point
fourth = (highest - average)/3 #4th edge/cut-off point
fifth = 2*((highest - average)/3) #5th edge/cut-off point

#Converting continuous 'gross' data into categorical data
#Code retrieved from Stack Overflow
category = pd.cut(df1.gross, bins=
                  [lowest, second, third, fourth, fifth, highest], labels=
                  ['Low', 'Below Average','Average','Above Average','High'])
df1['gross'] = category

df1 = df1.dropna(axis=0)

#Reading the test data
test_df = pd.read_csv('november_movies.csv')

df2 = test_df.replace(0, np.nan).dropna(axis=0)

#Determining the 6 edges or cut-off points of 5 bins/groups for the continuous data
lowest1 = df2['gross'].min() #Minimum gross revenue from the data, the 1st edge/cut-off point
average1 = df2['gross'].mean() #Average gross revenue from the data
second1 = (average1 - lowest1)/3 #2nd edge/cut-off point
third1 = 2*((average1 - lowest1)/3) #3rd edge/cut-off point
highest1 = df2['gross'].max() #Maximum gross revenue from the data, the last or 6th edge/cut-off point
fourth1 = (highest1 - average1)/3 #4th edge/cut-off point
fifth1 = 2*((highest1 - average1)/3) #5th edge/cut-off point

#Converting continuous 'gross' data into categorical data
#Code retrieved from Stack Overflow
category1 = pd.cut(df2.gross, bins=
                  [lowest1, second1, third1, fourth1, fifth1, highest1], labels=
                  ['Low', 'Below Average','Average','Above Average','High'])
df2['gross'] = category1
df2 = df2.dropna(axis=0)

#Creating train and test
X_train = df1.iloc[:,[1,2,6,8]]
y_train = df1['gross']
X_test = df2.iloc[:,[1,2,6,8]]
y_test = df2['gross']

#Create SVM classifier with polynomial kernel function
poly = svm.SVC(kernel='poly')

#Training the models using the training sets
poly.fit(X_train, y_train)

#Predict the response for test dataset
poly_pred = poly.predict(X_test)

print('Accuracy Polynomial Kernel:', metrics.accuracy_score(y_test, poly_pred))
print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, poly_pred))
print("Classification Report:\n:",metrics.classification_report(y_test, poly_pred))
