#importing libraries
import numpy as np
from PIL import Image
import os
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn import preprocessing
from matplotlib import style
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

scalar = preprocessing.StandardScaler()

#stacking input images to a single matrix
directory = os.getcwd()
inp_arr = np.zeros(25600)

for filename in os.listdir(directory):
    
    temp_image = Image.open(filename)
    temp_image = temp_image.resize((160,160))
    temp_arr = np.array(temp_image)
    temp_arr = temp_arr.flatten()
    inp_arr = np.vstack((inp_arr,temp_arr))
    
inp_arr = inp_arr[1:]

#getting labels of output
directory = os.getcwd()
y_arr = np.zeros(1)
for filename in os.listdir(directory):
    y = int(filename[7:9])
    y_arr = np.vstack((y_arr,y))
y_arr = y_arr[1:]

#Creating new datasets for train and test
#1.concatenating image and labels
#dataset = np.array([])
dataset = np.concatenate((inp_arr,y_arr), axis = 1)

#2.shuffle the dataset to create ransomness
for i in range(10):
    np.random.shuffle(dataset)
	
#3.split into test and train data
train = dataset[:120]
test = dataset[120:]

#pca module

def pca(inp_arr):
#preprocess data
    mean = np.mean(inp_arr,axis = 0)

    for i in inp_arr:
        i -= mean
    inp_arr = scalar.fit_transform(inp_arr)

#calculating singular value decomposition
    P,eig_val,V = LA.svd(inp_arr.transpose())
    #P.shape
    return P
	
#preprocessing train and test data and expexted outcomes

train_x = train[:,:25600]
train_y = train[:,-1]

test_x = test[:,:25600]
test_y = test[:,-1]

#k-fold cross-validation module

def cross_validation(dataset, folds):
	df_split = np.array_split(train, 5)
	dataset_copy = list()
	accuracy_final = []
	c = []
	final = []
	k = [30,60,90]
	for i in range(-10,10,2):
		temp = pow(2,i)
		c.append(temp)
		
	for j in c:
		for t in k:
			temp = []
			s_c = 'c = ' + str(j)
			s_k = 'k = ' + str(t)
			temp.append(s_c)
			temp.append(s_k)
			
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
			
			
				y_train = train_cv[:,-1]
				#print(y_train.size)
				X_train = train_cv[:,:25600]
				#print(X_train.size)
			 
				y_test = test_cv.iloc[:,-1]
				X_test = test_cv[:,:25600]
				
				P = pca(train)
				pca_train = np.dot(X_train, P.real[:,:t])
				pca_test = np.dot(X_test, P.real[:,:t])
			
				svclassifier = SVC(kernel= 'linear',C = j)  
				svclassifier.fit(pca_train, y_train)
				predictions = svclassifier.predict(pca_test)  

				accuracy = accuracy_score(y_test, predictions)
				accuracy_vec.append(accuracy)
			accuracy_per_fold = sum(accuracy_vec) / len(accuracy_vec)
			s_acc = 'accuracy = ' + str(accuracy_per_fold)
			temp.append(s_acc)
			accuracy_final.append(temp)
	return accuracy_final
	
#performing 5 fold cross-validation on training set
cross_val = cross_validation(train_x, 5)
print(cross_val)

#from the output of cross validation the maximum value at c =  and k = 
c = 256
k = 90
P = pca(train_x)
pca_train = np.dot(train_x, P.real[:,:k])
pca_test = np.dot(test_x, P.real[:,:k])
svclassifier = SVC(kernel= 'linear', c = )
svclassifier.fit(pca_train, train_y)
predictions_test = svclassifier.predict(pca_test)
accuracy = accuracy_score(test_y, predictions_test)


