
# prepare data
	
#Adding another column to the data file wherein if the burn
#area is positive, then the column have value 1 else 0.
#This is done to make the data compatible to classification.
import pandas as pd
df_train = pd.read_csv('train.csv')
df_train.columns
Burn = pd.Series([])
for i in range(len(df_train)):
	if df_train["area"][i] == 0:
		Burn[i]=0
	else: 
		Burn[i]=1
df_train.head()
df_train.insert(13, "Burn", Burn)
df_train.head()
df_train.to_csv('train.csv', index=False)

#Normalising the dataset

def zscore(X):
    my_data = (X - X.mean())/X.std()
    return my_data 
df_train1 = zscore(df_train)
df_train1.iloc[:,13] = df_train.iloc[:,13]
df_train1.head()

#Dividing dataset to train and test
train=df_train.sample(frac=0.8,random_state=200)
test=df_train.drop(train.index)
print ('Train set: ',len(train))
print ('Test set: ',len(test))

#Plotting data and inspecting whether the feature is
#catagorical or continuous.

#Scatter Plot between all the "features vs area"

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from plt.matplotlib import style


def plotFeatures(col_list,title):
	plt.figure(figsize=(10, 14))
	i = 0
	print(len(col_list))
	for col in col_list:
		i+=1
		plt.subplot(7,2,i)
		plt.plot(train[col],train["area"],marker='.',linestyle='none')
		plt.plot(train[col],np.log(train["area"]),marker='.',linestyle='none')
		plt.title(title % (col))   
		plt.tight_layout()

plotFeatures(train.columns,"Relationship bw %s and area")

#Correaltion heat map

import seaborn as sns
fig = plt.subplots(figsize = (10,10))
sns.set(font_scale=1.5)
sns.heatmap(train.corr(),square = True,cbar=True,annot=True,annot_kws={'size': 10})
plt.show()

#Calculating euclidean distance for our dataset which excludes the first 
#column and also the "area" and "burn" column.

import math
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1.iloc[x] - instance2.iloc[x]), 2)
	return math.sqrt(distance)
	
#Finding the K-Nearest Neighbours.

import operator 
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)
		distances.append((trainingSet.iloc[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
	
#Voting between the neighbours resulting in prediction of forest fire.

import operator
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

#Calculating accuracy of the prediction.

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		#print(testSet.iloc[x,-1],'and',predictions[x])
		if testSet.iloc[x,-1] == predictions[x]:
			correct += 1
		#print('correct :' ,correct)
	return (correct/float(len(testSet))) * 100.0
	

#Cross validation module for tuning k(hyperparameter)


def cross_validation(dataset, folds=5):
	df_split = np.array_split(train, 5)
	dataset_copy = list()
	accuracy_final = []
	for j in range(1,round(math.sqrt(len(train)))):
		for i in range(folds):
			fold = list()
			for fold_no in range(folds):
				if fold_no == i:
					test_cv = df_split[fold_no]
				else:
					fold.append(df_split[fold_no])
			train_cv = pd.concat(fold)
			print(len(train_cv))
			print(len(test_cv))
			accuracy_vec=[]
			predictions=[]
			for x in range(len(test_cv)):
				neighbors = getNeighbors(train_cv, test_cv.iloc[x], j)
				result = getResponse(neighbors)
				predictions.append(result)
				#print('> predicted=' ,(result) , ', actual=', (test_cv.iloc[x,-1]))
			accuracy = getAccuracy(test_cv, predictions)
			#print('Accuracy: ',(accuracy) , '%')
			accuracy_vec.append(accuracy)
		accuracy_per_fold = sum(accuracy_vec) / len(accuracy_vec)
		accuracy_final.append(accuracy_per_fold)
	return accuracy_final
a = cross_validation(df_train, folds=5)
a

#plotting graph for "accuracy vs k"

b = np.linspace(1,18,18)
pyplot.plot(b,a)
pyplot.show()

# training and testing model and generate predictions.
predictions=[]
k = 6
for x in range(len(test)):
	neighbors = getNeighbors(train, test.iloc[x], k)
	result = getResponse(neighbors)
	predictions.append(result)
	print('> predicted=' ,(result) , ', actual=', (test.iloc[x,-1]))
accuracy = getAccuracy(test, predictions)
print('Accuracy: ',(accuracy) , '%')

























