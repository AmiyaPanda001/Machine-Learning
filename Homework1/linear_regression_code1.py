
#Importing dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("train.csv")

#Deleting rows from dataset with burn area = 0
 
for i in range(len(dataset)):
	if dataset["area"][i] == 0:
		dataset = dataset.drop([i])
	else: 
		continue

#saving to the csv file

dataset.head()
dataset.to_csv('train.csv', index=False)

#plotting histogram of "burn area" in logarithmic scale

dataset.hist(column = 'area')
plt.hist(dataset['area'], log=True)

#Normalising the dataset

def zscore(X):
    my_data = (X - X.mean())/X.std()
    return my_data 
dataset = zscore(dataset)
dataset.head()

#Dividing dataset in test and train datasets separately

train=dataset.sample(frac=0.8,random_state=200)
test=dataset.drop(train.index)
X = train.iloc[:, :-2].values
y = train.iloc[:, 12].values

#Error metric function

def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))
	

#Covariance function

def cov_matrix(_y, _x): 
    if _x.shape[0] != _y.shape[0]:
        raise Exception("Shapes do not match")

    # make sure we use matrix multiplication, not array multiplication
    _xm = np.matrix(np.mean(_x, axis=0).repeat(_x.shape[0], axis = 0).reshape(_x.shape))
    _ym = np.matrix(np.mean(_y, axis=0).repeat(_y.shape[0], axis = 0).reshape(_y.shape))

    return ((_x - _xm).T * (_y - _ym)) * 1 / _x.shape[0]


#OLS calculating function

def compute_b0_bn(ym, Xm):
    
    if ym.shape[1] != 1:
        raise Exception ("ym should be a vector with shape [n, 1]")
        
    if Xm.shape[0] != ym.shape[0]:
        raise Exception ("Xm should have the same amount of lines as ym")
    
    C_y_x = cov_matrix(ym, Xm)
    C_x_x = cov_matrix(Xm, Xm)

    b1_bn  = C_x_x.I * C_y_x
    
    
    x_mean  = np.matrix(np.mean(Xm, axis = 0))
    y_mean  = np.mean(ym)
    
    rss = computeCost(Xm,ym,b1_bn.T)
    print(rss)
    b0 = -x_mean * b1_bn + y_mean
    
   # return (np.float(b0), np.array(b1_bn).flatten())
    #return rss
    return b1_bn


#OLS calculation

Xm = np.matrix(X)
ym = np.matrix(y.reshape((y.shape[0], 1)))

ret = compute_b0_bn(ym, Xm)
ret

X_test = test.iloc[:, :-2].values
y_test = test.iloc[:, 12].values
ret_test = computeCost(X_test,y_test,ret.T)
ret_test

test_output = X_test @ ret	

print(test_output)
print(y_test)

#Calculating co-relation between model ouptput and actual output 
 
test_output = test_output.getA1()
Covariance_test_output = cov_matrix(test_output, y_test)