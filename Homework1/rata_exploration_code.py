
# prepare data
	
#Delsting rows from the data file wherein if the burn
#area is equal to 0.
#This is done to make the data compatible to regression to find the burn area.
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


df_train = pd.read_csv('train.csv')
df_train.columns

for i in range(len(df_train)):
	if df_train["area"][i] == 0:
		df_train = df_train.drop([i])
	else: 
		continue

df_train.head()
df_train.to_csv('train.csv', index=False)

#Dividing dataset to train and test
train=df_train.sample(frac=0.8,random_state=200)
test=df_train.drop(train.index)

#Plotting data and inspecting whether the feature is
#catagorical or continuous.

#Scatter Plot between all the "features vs area"


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

#Normalisation function

def zscore(X):
    my_data = (X - X.mean())/X.std()
    return my_data 
dataset = zscore(dataset)




