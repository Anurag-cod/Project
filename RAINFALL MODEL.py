#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df= pd.read_csv("weatherAUS.csv")
df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


sns.heatmap(df.isnull(),yticklabels=False,cmap="viridis")


# In[7]:


#as evaporation,sunshine,cloud 9am,cloud 3pm,pressure 9am,pressure 3pm columns has lot of null values we can drop 
df.drop(["Evaporation","Sunshine","Pressure9am","Pressure3pm","Cloud9am","Cloud3pm","Date","Location"],inplace=True,axis=1)
df.head()


# In[8]:


df.shape


# In[9]:


sns.countplot(x="RainToday",data=df,palette='coolwarm')


# In[10]:


#dropping those ros which has null values
df.dropna(axis=0,inplace=True)
df.shape


# In[11]:


df.isnull().sum()


# In[12]:


df.columns


# In[13]:


#now our dataset has no null values as shown by heatmap
sns.heatmap(df.isnull(),yticklabels=False,cmap="viridis")


# # convert categorical features into dummy variable using pandas
# 

# In[14]:


Raintoday=pd.get_dummies(df["RainToday"],drop_first= True)
Raintoday.head()


# In[15]:


df=pd.concat([df,Raintoday],axis=1)
df.head()


# In[16]:


df.drop("RainToday",axis=1,inplace=True)
df.head()


# # LABEL ENCODING

# In[17]:


#ENCODING STRING TYPE DATA INTO NUMERICAL DATA
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["WindGustDir"]=le.fit_transform(df["WindGustDir"])
df["WindDir9am"]=le.fit_transform(df["WindDir9am"])
df["WindDir3pm"]=le.fit_transform(df["WindDir3pm"])
df["RainTomorrow"]=le.fit_transform(df["RainTomorrow"])


# In[18]:


df.head()


# In[22]:


df["WindGustDir"]


# # DATA VISULIZATION 

# In[19]:


#histogram for rain today
sns.distplot(df["Yes"],kde=False,bins=30)


# In[20]:


sns.countplot(x="WindGustDir",data=df)


# In[21]:


sns.countplot(x="WindDir9am",data=df,palette="winter")


# In[22]:


#scatter plot between Humidity9am and Temp9am
plt.figure(figsize = (8,8))
sns.scatterplot(x = 'Humidity9am', y = 'Temp9am', hue = 'RainTomorrow' , palette = 'inferno',data = df)


# In[23]:


plt.figure(figsize = (8,8))
sns.heatmap(df.corr(),lw=1)


# In[24]:


x=df.drop('RainTomorrow',axis=1)
y=df['RainTomorrow']


# # SPLITTING THE TRAIN AND TEST DATA

# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)


# # 1. Naive Bayes Classifier

# In[26]:


from sklearn.naive_bayes import GaussianNB
import pandas as pd

# Create the Naive Bayes model
model = GaussianNB()

# Train the model using the training data
model.fit(x_train,y_train)

# Use the model to predict tomorrow's rainfall
tomorrow_weather = pd.DataFrame({'MinTemp': [13], 'MaxTemp': [23], 'Rainfall': [1.0], 'WindGustDir' : [10], 'WindGustSpeed': [40],
       'WindDir9am': [10], 'WindDir3pm': [10], 'WindSpeed9am': [20.0], 'WindSpeed3pm' : [20],
       'Humidity9am': [70.0], 'Humidity3pm': [20.0], 'Temp9am': [17], 'Temp3pm': [20], 'RainToday': [0]})

predicted_rainfall = model.predict(tomorrow_weather)
print('The predicted rainfall for tomorrow is:', predicted_rainfall)

predicted_rainfall_prob = model.predict_proba(tomorrow_weather)
print ('The probability of rainfall for tomorrow is:',predicted_rainfall_prob)


# # 2.LOGISTIC REGRESSION MODEL
# 

# ###Training and Predicting

# In[33]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

# Scale the input data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Create and fit the logistic regression model
LR = LogisticRegression(max_iter=1000)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    LR.fit(x_train, y_train)

# Make predictions on the test data
predictions = LR.predict(x_test)


# In[29]:


#from sklearn.linear_model import LogisticRegression


# In[34]:


#LR = LogisticRegression()
#LR.fit(x_train,y_train)
#predictions = LR.predict(x_test)


# ##model Evaluation

# In[35]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("classification report is:")
print(classification_report(y_test,predictions))
print('\n')
print("confusion matrix is:")
print(confusion_matrix(y_test,predictions))
print('\n')
print("accuracy is:")
print(accuracy_score(y_test,predictions))


# In[41]:


from sklearn.datasets import load_iris
from sklearn.inspection import permutation_importance
data = load_iris()
clf = LogisticRegression()
clf.fit(x_train, y_train)

result = permutation_importance(clf, x_train, y_train, n_repeats=10, random_state=0)
importance = result.importances_mean

# Print feature importances
for i in range(len(importance)):
    print(f"Feature {i}: {importance[i]}")
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# In[ ]:





# # 3. DECISION TREE CLASSIFIER 

# #Training and predicting

# In[40]:


from sklearn.tree import DecisionTreeClassifier
DT= DecisionTreeClassifier()
DT.fit(x_train,y_train)


# In[41]:


predictions=DT.predict(x_test)


# #model Evaluation

# In[42]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("classification report is:")
print(classification_report(y_test,predictions))
print('\n')
print("confusion matrix is:")
print(confusion_matrix(y_test,predictions))
print('\n')
print("accuracy is:")
print(accuracy_score(y_test,predictions))


# In[52]:


# Get the feature importances
importance = DT.feature_importances_

# Print the feature importances
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

# Plot the feature importances
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# # 4.RANDOM FOREST CLASSIFIER

# #training and predicting

# In[43]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(x_train,y_train)


# In[44]:


predictions= RF.predict(x_test)


# model evaluation

# In[45]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("classification report is:")
print(classification_report(y_test,predictions))
print('\n')
print("confusion matrix is:")
print(confusion_matrix(y_test,predictions))
print('\n')
print("accuracy is:")
print(accuracy_score(y_test,predictions))


# In[54]:


# Get the feature importances
importance = RF.feature_importances_

# Print the feature importances
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

# Plot the feature importances
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# # 5.Ensemble Learning

# In[46]:


# Importing required libraries

from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Initializing the models
linear_reg = LinearRegression()
tree_reg = DecisionTreeRegressor()
forest_reg = RandomForestRegressor()

# Initializing the ensemble model
ensemble = VotingRegressor([('lr', linear_reg), ('dt', tree_reg), ('rf', forest_reg)])

# Fitting the ensemble model on the training data
ensemble.fit(x_train,y_train)

# Making predictions on the test data
y_pred = ensemble.predict(x_test)

# Calculating the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)


# In[ ]:




