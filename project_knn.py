####Amitabh Chakravorty
####Project: kNN

import pandas as pd
import statistics
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# new since our last experience with KNN
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

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
                  [lowest, third, fourth, highest], labels=
                  ['Below Average','Average','Above Average']) #Categorising into 3 groups since the test data is small
df1['gross'] = category

df1 = df1.dropna(axis=0)

#Reading the test data
test_df = pd.read_csv('november_movies.csv')

df2 = test_df.replace(0, np.nan).dropna(axis=0)
print(df2['gross'], "\n") #Printing out the numeric gross revenue values

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
                  [lowest1, third1, fourth1, highest1], labels=
                  ['Below Average','Average','Above Average']) #Categorising into 3 groups since the test data is small
df2['gross'] = category1
print(df2['gross'], "\n") #Printing out the nominal gross revenue values

df2 = df2.dropna(axis=0)

#Creating train and test
X_train = df1.iloc[:,[1,2,6,8]]
y_train = df1['gross']
X_test = df2.iloc[:,[1,2,6,8]]
y_test = df2['gross']

#Creating KNN Classfier model
knn = KNeighborsClassifier(n_neighbors=3)

#Fitting the training data
knn.fit(X_train,y_train)

#Predicting on the test data
pred = knn.predict(X_test)


print("Revenue categories predicted by the kNN model:\n", pred, "\n")
#Printing Confusion matrix, classification report and accuracy report 
print("Confusion matrix: ",confusion_matrix(y_test,pred), "\n")
print("Classification report", classification_report(y_test,pred), "\n")
print("Accuracy of the kNN model ",accuracy_score(y_test,pred))