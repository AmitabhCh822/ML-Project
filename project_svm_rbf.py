####Amitabh Chakravorty
####Project: SVM RBF

import pandas as pd
import numpy as np
import statistics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

#Reading the penguins data
df = pd.read_csv('penguins_size.csv')

#Removing all rows with missing values
df1 = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0) #Code retrieved from Stack Overflow


#Creating X and Y
X = df1.iloc[:, [2,3]] #Predictor or input variables are the culmen length and culmen depth
y = df1.iloc[:, 0] #Output variable is the species of penguin

#Normalizing the predictor attribute values
X_normalized = X.apply(lambda x: (x -min(x))/(max(x)-min(x)))

#Encoding strings to integer
y = LabelEncoder().fit_transform(y)

#Creating the training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) #70/30 train/test split

#Create SVM classifier with rbf kernel function
rbf = svm.SVC(kernel='rbf')

#Training the models using the training sets
rbf.fit(X_train, y_train)

#Predict the response for test dataset
rbf_pred = rbf.predict(X_test)

print("Revenue categories predicted by the svm model:\n", rbf_pred, "\n")
print('Accuracy of Radial Basis Kernel:', metrics.accuracy_score(y_test, rbf_pred), "\n")
print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, rbf_pred), "\n")
print("Classification Report:\n:",metrics.classification_report(y_test, rbf_pred))
