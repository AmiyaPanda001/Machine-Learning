# prepare data
	
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#Importing data

df_train = pd.read_csv('train.csv')
df_train.columns

#Counts number of Benign and Malignant cases

def count_B_M(df_train):
	count_B = 0
	count_M = 0
	for i in range(len(df_train)):
		if df_train["Result"][i] == 2:
			count_B += 1
		else: 
			count_M += 1
	return [count_B,count_M]


#Dividing dataset to train and test

train=df_train.sample(frac=0.67,random_state=200)
test=df_train.drop(train.index)

#Finding overall split of Benign and Malignant cases

total_split = count_B_M(df_train)
total_split
# result -------->  [444, 239]

#Finding split of Benign and Malignant cases on training data

count_B = 0
count_M = 0
for i in range(len(train)):
	if df_train["Result"][i] == 2:
		count_B += 1
	else: 
		count_M += 1
count_train = [count_B, count_M]
count_train
# result -------->  [271, 187]

#Finding split of Benign and Malignant cases on testing data

count_B = 0
count_M = 0
for i in range(len(test)):
	if df_train["Result"][i] == 2:
		count_B += 1
	else: 
		count_M += 1
count_test = [count_B, count_M]
count_test
# result -------->  [124, 101]

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
	

# Calculate the Gini index for a split dataset
def gini_index(groups,classes = [2,4]):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = list(group.iloc[:,-1]).count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Calculate the entropy index for a split dataset
	
def entropy(groups, classes = [2,4]):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted entropy for each group
	H = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = list(group.iloc[:,-1]).count(class_val) / size
			score += p * (-np.log(p))
		# weight the group score by its relative size
		H += (score) * (size / n_instances)
	return H

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in range(len(dataset)):
		if dataset.iloc[row,index] < value:
			left.append(dataset.iloc[row,:])
		else:
			right.append(dataset.iloc[row,:])
	df_left = pd.DataFrame(left)
	df_right = pd.DataFrame(right)
	return [df_left, df_right]
	
# gettting the best split

def get_split(dataset,metric):
	class_values = list(set(dataset.iloc[:,-1]))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset.columns)-1):
		for row in range(len(dataset)):
			groups = test_split(index, dataset.iloc[row,index], dataset)
			if metric == 'gini':
				score = gini_index(groups, class_values)
			else:
				score = entropy(groups, class_values)
			if score < b_score:
				b_index, b_value, b_score, b_groups = index, dataset.iloc[row,index], score, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
	

# Create a terminal node value
def to_terminal(group):
	outcomes = list(group.iloc[:,-1]);#print(outcomes)
	return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth, metric ):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if left.empty or right.empty:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left,metric)
		split(node['left'], max_depth, min_size, depth+1, metric)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right,metric)
		split(node['right'], max_depth, min_size, depth+1,metric)
		

#Build decision tree				
def build_tree(train, max_depth, min_size,metric):
	root = get_split(train,metric)
	split(root, max_depth, min_size, 1,metric)
	return root


# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size,metric):
	tree = build_tree(train, max_depth, min_size,metric)
	predictions = list()
	for row in range(len(test)):
		prediction = predict(tree, test.iloc[row])
		predictions.append(prediction)
	return(predictions)

# Calculating accuracy for different levels of tree formation 

# Gini index based tree

accuracy_vec = []

for i in range(20):
	prediction = []
	prediction = decision_tree(train,test,i,1,'gini')
	accuracy = accuracy_metric(list(test.iloc[:,-1]), prediction)
	accuracy_vec.append(accuracy)
	
accuracy_vec = []

# Entropy based tree

for i in range(20):
	prediction = []
	prediction = decision_tree(train,test,i,1,'gini')
	accuracy = accuracy_metric(list(test.iloc[:,-1]), prediction)
	accuracy_vec.append(accuracy)
	