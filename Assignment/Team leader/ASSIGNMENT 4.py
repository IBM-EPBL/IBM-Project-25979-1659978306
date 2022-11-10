#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[12]:


import os
os.chdir("./desktop/")


# In[13]:


abalone = pd.read_csv('abalone.csv')


# In[14]:


abalone


# In[15]:


abalone.head()


# In[16]:


abalone.tail()


# In[17]:


##age can be calculated by using value 1.5 to rings 


# In[23]:


abalone['age'] = abalone['Rings']+1.5
abalone = abalone.drop ('Rings',axis=1)


# # Univariate analysis

# In[24]:


sns.heatmap(abalone.isnull())


# In[27]:


plt.figure(figsize = (20,7))
sns.swarmplot(x = 'Sex', y = 'age', data = abalone, hue = 'Sex')
sns.violinplot(x = 'Sex', y = 'age',data = abalone)


# In[26]:


sns.countplot(x = 'Sex', data = abalone, palette = 'Set3')


# # Bivariate Analysis

# In[ ]:


numerical_features = abalone.select_dtypes(include = [np.number]).columns
categorical_features = abalone.select_dtypes(include = [np.object]).column


# In[30]:


numerical_features


# In[31]:


categorical_features


# In[32]:


plt.figure(figsize = (20,7))
sns.heatmap(abalone[numerical_features].corr(),annot = True)


# # Multivariate Analysis
# 

# In[33]:


sns.pairplot(abalone)


# # Descriptive Statistics

# In[34]:


#CONTINUOUS VARIABLES


# In[35]:


abalone['Length'].describe()


# In[36]:


abalone['Shucked weight'].describe()


# In[37]:


abalone['Shell weight'].describe()


# In[38]:


abalone['Height'].describe()


# In[39]:


# Categorical variable


# In[40]:


abalone['Sex'].describe()


# In[41]:


abalone['Sex'].value_counts()


# In[42]:


#Distribution measures


# In[44]:


abalone['Length'].kurtosis()


# In[45]:


abalone['Length'].skew()


# In[46]:


abalone['Shucked weight'].kurtosis()


# In[47]:


abalone['Shucked weight'].skew()


# # Missing values

# In[48]:


missing_values = abalone.isnull().sum()


# In[49]:


missing_values


# In[51]:


missing_values = abalone.isnull().sum().sort_values(ascending = False)
percentage_missing_values = (missing_values/len(abalone))*100
pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing'])


# # Outliers

# In[52]:


abalone = pd.get_dummies(abalone)
dummy_df = abalone


# In[53]:


abalone.boxplot( rot = 90, figsize=(20,5))


# In[54]:


var = 'Viscera weight'
plt.scatter(x = abalone[var], y = abalone['age'])
plt.grid(True)


# In[55]:


abalone.drop(abalone[(abalone['Viscera weight']> 0.5) & (abalone['age'] < 20)].index, inplace=True)
abalone.drop(abalone[(abalone['Viscera weight']<0.5) & (abalone['age'] > 25)].index, inplace=True)


# In[56]:


var = 'Shucked weight'
plt.scatter(x = abalone[var], y = abalone['age'])
plt.grid(True)


# In[57]:


abalone.drop(abalone[(abalone['Viscera weight']> 0.5) & (abalone['age'] < 20)].index, inplace=True)
abalone.drop(abalone[(abalone['Viscera weight']<0.5) & (abalone['age'] > 25)].index, inplace=True)


# In[58]:


var = 'Shell weight'
plt.scatter(x = abalone[var], y = abalone['age'])
plt.grid(True)


# In[59]:


abalone.drop(abalone[(abalone['Shell weight']> 0.6) & (abalone['age'] < 25)].index, inplace=True)
abalone.drop(abalone[(abalone['Shell weight']<0.8) & (abalone['age'] > 25)].index, inplace=True)


# In[60]:


var = 'Shucked weight'
plt.scatter(x = abalone[var], y = abalone['age'])
plt.grid(True)


# In[61]:


abalone.drop(abalone[(abalone['Shucked weight'] >= 1) & (abalone['age'] < 20)].index, inplace = True)
abalone.drop(abalone[(abalone['Viscera weight']<1) & (abalone['age'] > 20)].index, inplace = True)


# In[62]:


var = 'Whole weight'
plt.scatter(x = abalone[var], y = abalone['age'])
plt.grid(True)


# In[63]:


abalone.drop(abalone[(abalone['Whole weight'] >= 2.5) & (abalone['age'] < 25)].index, inplace = True)
abalone.drop(abalone[(abalone['Whole weight']<2.5) & (abalone['age'] > 25)].index, inplace = True)


# In[64]:


var = 'Diameter'
plt.scatter(x = abalone[var], y = abalone['age'])
plt.grid(True)


# In[65]:


abalone.drop(abalone[(abalone['Diameter'] <0.1) & (abalone['age'] < 5)].index, inplace = True)
abalone.drop(abalone[(abalone['Diameter']<0.6) & (abalone['age'] > 25)].index, inplace = True)
abalone.drop(abalone[(abalone['Diameter']>=0.6) & (abalone['age'] < 25)].index, inplace = True)


# In[66]:


var = 'Height'
plt.scatter(x = abalone[var], y = abalone['age'])
plt.grid(True)


# In[67]:


abalone.drop(abalone[(abalone['Height'] > 0.4) & (abalone['age'] < 15)].index, inplace = True)
abalone.drop(abalone[(abalone['Height']<0.4) & (abalone['age'] > 25)].index, inplace = True)


# In[68]:


var = 'Length'
plt.scatter(x = abalone[var], y = abalone['age'])
plt.grid(True)


# In[69]:


abalone.drop(abalone[(abalone['Length'] <0.1) & (abalone['age'] < 5)].index, inplace = True)
abalone.drop(abalone[(abalone['Length']<0.8) & (abalone['age'] > 25)].index, inplace = True)
abalone.drop(abalone[(abalone['Length']>=0.8) & (abalone['age'] < 25)].index, inplace = True)


# In[70]:


abalone


# # Categorical columns

# In[ ]:


numerical_features = abalone.select_dtypes(include = [np.number]).columns
categorical_features = abalone.select_dtypes(include = [np.object]).columns


# In[72]:


numerical_features


# In[73]:


categorical_features


# In[75]:


abalone_numeric = abalone[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight','Viscera weight', 'Shell weight', 'age', 'Sex_F', 'Sex_I', 'Sex_M']]


# In[76]:


abalone_numeric.head()


# # Dependent and Independent Variables

# In[77]:


x = abalone.iloc[:, 0:1].values


# In[78]:


y = abalone.iloc[:, 1]


# In[79]:


x


# In[81]:


y


# # Scaling the Independent Variables

# In[82]:


print ("\n ORIGINAL VALUES: \n\n", x,y) 


# In[83]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1)) 
new_y= min_max_scaler.fit_transform(x,y) 
print ("\n VALUES AFTER MIN MAX SCALING: \n\n", new_y)


# # Split the data into Training and Testing

# In[84]:


X = abalone.drop('age', axis = 1)
y = abalone['age']


# In[85]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest


# In[86]:


standardScale = StandardScaler()
standardScale.fit_transform(X)

selectkBest = SelectKBest()
X_new = selectkBest.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.25)


# # Build the model

# # Linear Regression

# In[87]:


from sklearn import linear_model as lm
from sklearn.linear_model import LinearRegression
model=lm.LinearRegression()
results=model.fit(X_train,y_train) 


# In[88]:


accuracy = model.score(X_train, y_train)
print('Accuracy of the model:', accuracy)


# In[89]:


lm = LinearRegression()
lm.fit(X_train, y_train)


# In[90]:


y_train_pred = lm.predict(X_train)
y_test_pred = lm.predict(X_test)


# # Training the model

# In[91]:


X_train


# In[92]:


y_train


# In[93]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
s = mean_squared_error(y_train, y_train_pred)
print('Mean Squared error of training set :%2f'%s)


# # Testing the model

# In[95]:


X_test


# In[96]:


y_test


# In[97]:


p = mean_squared_error(y_test, y_test_pred)
print('Mean Squared error of testing set :%2f'%p)


# In[98]:


from sklearn.metrics import r2_score
s = r2_score(y_train, y_train_pred)
print('R2 Score of training set:%.2f'%s)


# In[99]:


from sklearn.metrics import r2_score
p = r2_score(y_test, y_test_pred)
print('R2 Score of testing set:%.2f'%p)


# In[ ]:




