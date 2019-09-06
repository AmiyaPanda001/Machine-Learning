#importing libraries
import numpy as np
from PIL import Image
import os
from numpy import linalg as LA
#import matplotlib.pyplot as plt
from sklearn import preprocessing

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

#1.concatenating image and labels
#dataset = np.array([])
dataset = np.concatenate((inp_arr,y_arr), axis = 1)

#2.shuffle the dataset to create randomness
for i in range(10):
    np.random.shuffle(dataset)
	
#3.split into test and train data
train = dataset[:120]
test = dataset[120:]

train_x = train[:,:25600].reshape((120,160,160,1))
train_y = train[:,-1]

test_x = test[:,:25600].reshape((45,160,160,1))
test_y = test[:,-1]

#importing keras libraries
import keras
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPool2D
from keras.models import Sequential

#Preprocess image
train_x = train_x/255.
test_x = test_x/255.

#onehot encoding
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

#Model module--------------------------------------------------------------------------------------

act = ['relu','tanh']
drp = [0.2,0.3]
fil = [3,5]
strd = [1,2]
accuracy_vec = []

for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
#output---------------------------------------------------------------------------------------------                
                accuracy = []
                s_act = 'act = ' + str(act[i])
                s_drp = 'drp = ' + str(drp[j])
                s_fil = 'fil = ' + str(fil[k])
                s_strd = 'strd = ' + str(strd[l])
                accuracy.append(s_act)
                accuracy.append(s_drp)
                accuracy.append(s_fil)
                accuracy.append(s_strd)
#Model description----------------------------------------------------------------------------------

                model = Sequential()
                model.add(Conv2D(32, kernel_size = fil[k],strides= strd[l], activation=act[i], input_shape=(160,160,1)))
                model.add(MaxPool2D(2))
                model.add(Conv2D(64, kernel_size= fil[k],strides=strd[l],  activation=act[i]))
                model.add(MaxPool2D(2))
                model.add(Conv2D(128, kernel_size= fil[k],strides= strd[l],  activation=act[i]))
                model.add(MaxPool2D(2))
                model.add(Flatten())
                model.add(Dropout(drp[j]))
                model.add(Dense(16, activation='softmax'))

                optim = keras.optimizers.Adam(lr=0.001)
#compile model-----------------------------------------------------------------------------------------
                model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
#fit model---------------------------------------------------------------------------------------------
                model.fit(train_x,train_y, epochs=8, batch_size=32)
#evaluate model----------------------------------------------------------------------------------------
                loss,acc = model.evaluate(test_x, test_y)
#output------------------------------------------------------------------------------------------------    
                s_acc = 'accuracy = ' + str(acc)
                accuracy.append(s_acc)
                accuracy_vec.append(accuracy)

print(accuracy_vec)

#Data Augmentation module-------------------------------------------------------------------------

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False)

datagen.fit(train_x)

#Again Running the model with augmented data-------------------------------------------------------

act = ['relu','tanh']
drp = [0.2,0.3]
fil = [3,5]
strd = [1,2]
accuracy_vec = []

for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
#output---------------------------------------------------------------------------------------------                
                accuracy = []
                s_act = 'act = ' + str(act[i])
                s_drp = 'drp = ' + str(drp[j])
                s_fil = 'fil = ' + str(fil[k])
                s_strd = 'strd = ' + str(strd[l])
                accuracy.append(s_act)
                accuracy.append(s_drp)
                accuracy.append(s_fil)
                accuracy.append(s_strd)
#Model description----------------------------------------------------------------------------------

                model = Sequential()
                model.add(Conv2D(32, kernel_size = fil[k],strides= strd[l], activation=act[i], input_shape=(160,160,1)))
                model.add(MaxPool2D(2))
                model.add(Conv2D(64, kernel_size= fil[k],strides=strd[l],  activation=act[i]))
                model.add(MaxPool2D(2))
                model.add(Conv2D(128, kernel_size= fil[k],strides= strd[l],  activation=act[i]))
                model.add(MaxPool2D(2))
                model.add(Flatten())
                model.add(Dropout(drp[j]))
                model.add(Dense(16, activation='softmax'))

                optim = keras.optimizers.Adam(lr=0.001)
#compile model-----------------------------------------------------------------------------------------
                model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
#fit model---------------------------------------------------------------------------------------------
                model.fit(train_x,train_y, epochs=8, batch_size=32)
#evaluate model----------------------------------------------------------------------------------------
                loss,acc = model.evaluate(test_x, test_y)
#output------------------------------------------------------------------------------------------------    
                s_acc = 'accuracy = ' + str(acc)
                accuracy.append(s_acc)
                accuracy_vec.append(accuracy)
				
print(accuracy_vec)