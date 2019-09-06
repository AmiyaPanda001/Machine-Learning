#importing libraries
import numpy as np
from PIL import Image
import os
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn import preprocessing

scalar = preprocessing.StandardScaler()

#stacking input images to a single matrix
#make sure to be in same directory as the images

directory = os.getcwd()
inp_arr = np.zeros(25600)

for filename in os.listdir(directory):
    
    temp_image = Image.open(filename)
    temp_image = temp_image.resize((160,160))
    temp_arr = np.array(temp_image)
    temp_arr = temp_arr.flatten()
    inp_arr = np.vstack((inp_arr,temp_arr))
    
inp_arr = inp_arr[1:]
mean = np.mean(inp_arr,axis = 0)

for i in inp_arr:
    i -= mean
inp_arr = scalar.fit_transform(inp_arr)

#calculating singular value decomposition
P,eig_val,V = LA.svd(inp_arr.transpose())

print(P.shape)

#getting labels of output
directory = os.getcwd()
y_arr = np.zeros(1)
for filename in os.listdir(directory):
    y = int(filename[7:9])
    y_arr = np.vstack((y_arr,y))
y_arr = y_arr[1:]

#calculating the cummulative energy per eigen vector no.(k) increase
energy_vec = []
tot_lamda = np.sum(eig_val)
for i in range(len(P)):
    temp_eig = eig_val[0:i]
    temp_lambda = np.sum(temp_eig)
    energy = (temp_lambda/tot_lamda)*100
    energy_vec.append(energy)
    
#printing intital 50 elements of energy_vec
print(energy_vec[:50])

#plotting graph for cummilative energy per k
temp_energy = energy_vec[1:25]
plt.plot(temp_energy)

#plotting top 10 eignfaces 
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(P.real[:,i].reshape((160,160)), cmap = plt.cm.gray)
    plt.title('k = {}'.format(i+1))
	
#Reconstruction of image from k eigenvectors

k = [1, 10, 20, 30, 40, 50, 100, 150]

for i in k:
    u = P.real[:,:i+1]
    temp = np.dot(inp_arr[100], P.real[:,:i+1])
    #print(temp.shape)
    temp = np.dot(u,temp) 
    plt.imshow(temp.reshape((160,160)), cmap = plt.cm.gray)
    plt.title('k = {}'.format(i))
    plt.show()
