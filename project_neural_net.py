####Amitabh Chakravorty
####Project: Neural Networks

import pandas as pd
import numpy as np
import statistics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

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
df2 = df2.dropna(axis=0)

#Creating train and test
X_train = df1.iloc[:,[1,2,6,8]].astype('float32') #Ensuring all data are floating point values
y_train = LabelEncoder().fit_transform(df1['gross']) #Encoding strings to integer
X_test = df2.iloc[:,[1,2,6,8]].astype('float32') #Ensuring all data are floating point values
y_test = LabelEncoder().fit_transform(df2['gross']) #Encoding strings to integer

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Determining the number of input features
n_features = X_train.shape[1]
#Defining model
model = Sequential()

#Adding layers
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(5, activation='softmax'))

#Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Fitting the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

model.summary()

#Evaluating the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)