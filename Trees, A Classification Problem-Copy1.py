
# coding: utf-8

# ## Trees: A Classification Problem
# 
# 
# **Using ML techniques to classify trees**
# - K-nearest Neighbors
# - Decision Trees
# - Neural Networks

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sk


# In[2]:


#Import the data Locally
data = pd.read_csv(r"C:\Users\bton\Desktop\Tree\Tree.csv")
data.head()


# In[3]:


data.tail()


# In[68]:


#shape of entire data set 14 features, 1 response)
data.shape


# In[4]:


#Separating the data set into features and response
features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']
X = data[features]
X = data [['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']]
X.head()


# In[69]:


print (X.shape)


# In[5]:


##Separating the data set into features and response
y = ['Cover_Type']
y = data.Cover_Type
y.head()


# In[6]:


print (y.shape)


# In[14]:


#Splitting dataset into training set and testing sets (3:1 split)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[15]:


#Print the shape of the four new datasets
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)


# In[16]:


#import K nearest neighbors module, k=1
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
print (knn)


# In[17]:


#Training the model
knn.fit(X_train, y_train)


# In[18]:


#Model on test data set
knn.predict(X_test)


# In[74]:


#Model Accuracy
y_pred = knn.predict(X_test)
len(y_pred)
from sklearn import metrics
print (metrics.accuracy_score(y_test, y_pred))


# In[75]:


#Build K nearest neighbors model, k=5; test accuracy
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, y_train)
knn5.predict(X_test)
y_pred5 = knn5.predict(X_test)
print (metrics.accuracy_score(y_test, y_pred5))


# In[76]:


#import K nearest neighbors module, k=10; test accuracy
knn10 = KNeighborsClassifier(n_neighbors=10)
knn10.fit(X_train, y_train)
knn10.predict(X_test)
y_pred10 = knn10.predict(X_test)
print (metrics.accuracy_score(y_test, y_pred10))


# In[98]:


#import K nearest neighbors module, k=10; test accuracy
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, y_train)
knn3.predict(X_test)
y_pred3 = knn3.predict(X_test)
print (metrics.accuracy_score(y_test, y_pred3))


# In[84]:


print (metrics.accuracy_score(y_test, y_pred))
print (metrics.accuracy_score(y_test, y_pred5))
print (metrics.accuracy_score(y_test, y_pred10))


# In[72]:


# try k=1 through k=15 to find the best k
k_range = range(1, 16)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    knn.predict(X_test)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
print (scores)


# In[73]:


#plot relationship between k and testing accuracy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy')


# In[47]:


#Decision Tree Classifiction Models
from sklearn.tree import DecisionTreeClassifier


# In[48]:


#Fit Regression Model
Tree1 = DecisionTreeClassifier(max_depth=10)
Tree1.fit(X_train, y_train)
Tree2 = DecisionTreeClassifier(max_depth=15)
Tree2.fit(X_train, y_train)
Tree3 = DecisionTreeClassifier(max_depth=25)
Tree3.fit(X_train, y_train)
Tree4 = DecisionTreeClassifier(max_depth=35)
Tree4.fit(X_train, y_train)


# In[51]:


#Accuracy of Trees
print('Accuracy on Training data set of Tree 1: {:.3f}'.format(Tree1.score(X_test, y_test)))
print('Accuracy on Training data set of Tree 2: {:.3f}'.format(Tree2.score(X_test, y_test)))
print('Accuracy on Training data set of Tree 3: {:.3f}'.format(Tree3.score(X_test, y_test)))
print('Accuracy on Training data set of Tree 4: {:.3f}'.format(Tree4.score(X_test, y_test)))


# In[87]:


Tree1 = DecisionTreeClassifier(criterion = "entropy", max_depth=10)
Tree1.fit(X_train, y_train)
Tree2 = DecisionTreeClassifier(criterion = "entropy", max_depth=15)
Tree2.fit(X_train, y_train)
Tree3 = DecisionTreeClassifier(criterion = "entropy", max_depth=25)
Tree3.fit(X_train, y_train)
Tree4 = DecisionTreeClassifier(criterion = "entropy", max_depth=35)
Tree4.fit(X_train, y_train)
#Accuracy of Trees
print('Accuracy on Training data set of Tree 1: {:.3f}'.format(Tree1.score(X_test, y_test)))
print('Accuracy on Training data set of Tree 2: {:.3f}'.format(Tree2.score(X_test, y_test)))
print('Accuracy on Training data set of Tree 3: {:.3f}'.format(Tree3.score(X_test, y_test)))
print('Accuracy on Training data set of Tree 4: {:.3f}'.format(Tree4.score(X_test, y_test)))


# In[95]:


#Random Forests
from sklearn.ensemble import RandomForestClassifier
RF1 = RandomForestClassifier(n_estimators=25, max_depth=35)
RF1.fit(X_train, y_train)
RF2 = RandomForestClassifier(n_estimators=50, max_depth=35)
RF2.fit(X_train, y_train)
RF3 = RandomForestClassifier(n_estimators=100, max_depth=35)
RF3.fit(X_train, y_train)
RF4 = RandomForestClassifier(n_estimators=25, criterion = "entropy", max_depth=35)
RF4.fit(X_train, y_train)
RF5 = RandomForestClassifier(n_estimators=50, criterion = "entropy", max_depth=35)
RF5.fit(X_train, y_train)
RF6 = RandomForestClassifier(n_estimators=100,criterion = "entropy", max_depth=35)
RF6.fit(X_train, y_train)
print('Accuracy on Training data set of RF1: {:.3f}'.format(RF1.score(X_test, y_test)))
print('Accuracy on Training data set of RF2: {:.3f}'.format(RF2.score(X_test, y_test)))
print('Accuracy on Training data set of RF3: {:.3f}'.format(RF3.score(X_test, y_test)))
print('Accuracy on Training data set of RF4: {:.3f}'.format(RF4.score(X_test, y_test)))
print('Accuracy on Training data set of RF5: {:.3f}'.format(RF5.score(X_test, y_test)))
print('Accuracy on Training data set of RF6: {:.3f}'.format(RF4.score(X_test, y_test)))


# In[100]:


#Neural Network
from sklearn.neural_network import MLPClassifier
NN = MLPClassifier(random_state=0)
NN.fit(X_train, y_train)
print('Accuracy on the test dataset: {:.3f}'.format(NN.score(X_test,y_test)))


# In[102]:


#Standardizing the Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.fit(X_test).transform(X_test)
NN = MLPClassifier(max_iter = 1000, random_state = 0)
NN.fit(X_train_scaled, y_train)
print('Accuracy on the test dataset: {:.3f}'.format(NN.score(X_test_scaled,y_test)))

