#data pre-processing
#importing packages

import numpy as np
import pandas as pd
from matplotlib import style
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time

#importing data

data = pd.read_csv('svm_train.csv', header=None)

#to find out columns with 3 variable counts

data.nunique()

#adding features 

col_list = [1, 6, 7, 13, 14, 15, 25, 28]

for i in col_list:
    a1 = pd.Series([])
    a2 = pd.Series([])
    a3 = pd.Series([])
    for j in range(len(data)):
        if data.iloc[j,i] == -1:
            a1[j] = 1
            a2[j] = 0
            a3[j] = 0
        elif data.iloc[j,i] == 0:
            a1[j] = 0
            a2[j] = 1
            a3[j] = 0
        elif data.iloc[j,i] == 1:
            a1[j] = 0
            a2[j] = 0
            a3[j] = 1
    data.insert(loc = len(data.columns),column = len(data.columns), value = a1)
    data.insert(loc = len(data.columns),column = len(data.columns), value = a2)
    data.insert(loc = len(data.columns),column = len(data.columns), value = a3)
	

data.head()

data.drop(col_list, inplace=True, axis=1)
len(data.columns)
data.columns

#changing -1 to 0 

for i in range(22):
    for j in range(len(data)):
        if data.iloc[j,i] == -1:
            data.iloc[j,i] = 0
        else:
            data.iloc[j,i] = 1

data.to_csv('train_1.csv', index=False)

#splitting test and train data

train=data.sample(frac=0.67,random_state=200)
test=data.drop(train.index)

#cross_validation 

def cross_validation(dataset, folds, kernel, gamma = 'auto'):
	df_split = np.array_split(train, 3)
	dataset_copy = list()
	accuracy_final = []
	c = []
	time_vec = []
	final = []
	for i in range(-10,10):
		temp = pow(2,i)
		c.append(temp)
		
	for j in c:
		t0 = time.time()
		for i in range(folds):
			fold = list()
			for fold_no in range(folds):
				if fold_no == i:
					test_cv = df_split[fold_no]
				else:
					fold.append(df_split[fold_no])
			train_cv = pd.concat(fold)
			accuracy_vec=[]
			predictions=[]
			
			
			y_train = train_cv.iloc[:,22]
			#print(y_train.size)
			X_train = train_cv.drop(columns = 30)
			#print(X_train.size)
			 
			y_test = test_cv.iloc[:,22]
			X_test = test_cv.drop(columns = 30)
			
			svclassifier = SVC(kernel= kernel,C = j, gamma = gamma)  
			svclassifier.fit(X_train, y_train)
			predictions = svclassifier.predict(X_test)  

			accuracy = accuracy_score(y_test, predictions)
			accuracy_vec.append(accuracy)
		accuracy_per_fold = sum(accuracy_vec) / len(accuracy_vec)
		accuracy_final.append(accuracy_per_fold)
		t1 = time.time()
		total = t1 - t0
		time_vec.append(total)
	final = zip(accuracy_final,time_vec)
	return final

###################################for linear kernel###################################

a = cross_validation(train, 3, 'linear')
list(a)

#considering C = 2^5 = 32 on test set
y = test.iloc[:,22]
X = test.drop(columns = 30)
predictions_test = []

#training 

y_train = train.iloc[:,22]
X_train = train.drop(columns = 30)

svclassifier = SVC(kernel= 'linear',C = 32)  
svclassifier.fit(X_train, y_train)

#prediction

predictions_test = svclassifier.predict(X)

accuracy = accuracy_score(y, predictions_test)
accuracy

###################################for polynimal kernel###################################

cv_poly = cross_validation(train, 3, 'poly')
list(cv_poly)

#considering C = 2^6 = 64 on test set
y = test.iloc[:,22]
X = test.drop(columns = 30)
predictions_test = []

#training 

y_train = train.iloc[:,22]
X_train = train.drop(columns = 30)

svclassifier = SVC(kernel= 'poly',C = 64)  
svclassifier.fit(X_train, y_train)

#prediction

predictions_test = svclassifier.predict(X)

accuracy = accuracy_score(y, predictions_test)
accuracy

###################################for rbf kernel###################################

cv_rbf = cross_validation(train, 3, 'rbf')
list(cv_rbf)

#considering C = 2^6 = 64 on test set
y = test.iloc[:,22]
X = test.drop(columns = 30)
predictions_test = []

#training 

y_train = train.iloc[:,22]
X_train = train.drop(columns = 30)

svclassifier = SVC(kernel= 'rbf',C = 64)  
svclassifier.fit(X_train, y_train)

#prediction

predictions_test = svclassifier.predict(X)

accuracy = accuracy_score(y, predictions_test)
accuracy






