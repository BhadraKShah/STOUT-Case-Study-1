#!/usr/bin/env python
# coding: utf-8

# # Case Study 1

# In[1]:


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[2]:


loan = pd.read_csv("loans_full_schema .csv")


# In[3]:


loan.shape


# This dataset comprises thousands of loans made using the Lending Club platform, which allows people to lend money to other people.
# The dataset has 10000 observations and 55 features. The number of features is large and the number of observations are less

# In[4]:


loan.dtypes


# In[5]:


loan.head()


# In[6]:


df1 = loan.loc[:,:'num_collections_last_12m']
df_f = loan['interest_rate']
heat1 = pd.concat([df1,df_f],axis = 1)
plt.figure(figsize=(15,10))
sb.heatmap(heat1.corr(),annot=True)
plt.show()


# In[7]:


df2 = loan.loc[:,'num_historical_failed_to_pay':'num_accounts_30d_past_due']
heat2 = pd.concat([df2,df_f],axis = 1)
plt.figure(figsize=(11,10))
sb.heatmap(heat2.corr(),annot=True)
plt.show()


# In[8]:


df4 = loan.loc[:,'num_active_debit_accounts':'term']
heat3 = pd.concat([df4,df_f],axis = 1)
plt.figure(figsize=(11,10))
sb.heatmap(heat3.corr(),annot=True)
plt.show()


# In[9]:


plt.figure(figsize=(8,8))
sb.heatmap(loan.loc[:,'interest_rate':].corr(),annot=True)
plt.show()


# In heat maps above we can see correlations of all the variables with respect to interest rate and as we can see almost all the features are positively correlated not not highly correlated.

# In[10]:


loan.describe()


# In[11]:


loan.isnull().sum()


# There are many colums with null values.

# In[12]:


for i in loan.columns:
    if(loan[i].dtypes == object):
        loan[i] = loan[i].astype('category')


# Converting all the features of object type to category

# In[13]:


loan = loan.drop(['emp_title','annual_income_joint','verification_income_joint','debt_to_income_joint','months_since_last_delinq','debt_to_income','months_since_90d_late','months_since_last_credit_inquiry','num_accounts_120d_past_due','emp_length'],axis = 1)


# dropping the variables with null values.
# 
# Other steps we can do is:
# - replace the values of joint income with normal income at null values
# - replace the other values with mean of the column

# In[14]:


loan.dtypes


# ### Data Visualizations

# In[15]:


plt.figure(figsize=(10,5))
sb.violinplot(x = loan['verified_income'],y=loan['interest_rate'])
plt.show()


# Here we can see the distribution of verified income and interest rate, this is an interesting observation because the average interest rate of not verified income is lower than the average rate of verified and source verified.

# In[16]:



import squarify 

st = loan['state'].value_counts()
plt.figure(figsize=(8,8))
squarify.plot(sizes=st, label=st.index, alpha=0.6 )
plt.show()


# This is a tree map which explains the number of candidates from each state we can see that the number of people from california are more than any other state.

# In[17]:


plt.figure(figsize=(15,5))
sb.stripplot(loan['num_total_cc_accounts'],loan['interest_rate'])
plt.show()


# This plot shows the interest rate w.r.t to the number of credit account a person holds, we can infer that credit account does not effect the interest rate.

# In[18]:


plt.figure(figsize=(10,8))
sb.scatterplot(x=loan['grade'],y=loan['interest_rate'])
plt.xlabel('grade')
plt.ylabel('Interest_rate')
plt.show()


# Depending on the grade the interest rate is clearly increasing 

# In[19]:


import numpy as np
int_rate = loan['interest_rate'].value_counts()
int_rate = int_rate.reset_index()

plt.figure(figsize=(20,10))


ax = plt.subplot(111, polar=True)


plt.axis('off')


upperLimit = 300
lowerLimit = 100


max = int_rate['interest_rate'].max()

slope = (max - lowerLimit) / max
heights = slope * int_rate.interest_rate + lowerLimit


width = 2*np.pi / len(int_rate.index)


indexes = list(range(1, len(int_rate.index)+1))
angles = [element * width for element in indexes]
angles


bars = ax.bar(
    x=angles, 
    height=heights, 
    width=width, 
    bottom=lowerLimit,
    linewidth=2, 
    edgecolor="white")


labelPadding = 4


for bar, angle, height, label in zip(bars,angles, heights, int_rate["index"]):

    
    rotation = np.rad2deg(angle)

    alignment = ""
    if angle >= np.pi/2 and angle < 3*np.pi/2:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"

 
    ax.text(
        x=angle, 
        y=lowerLimit + bar.get_height() + labelPadding, 
        s=label, 
        ha=alignment, 
        va='center', 
        rotation=rotation, 
        rotation_mode="anchor") 


# This plot shows the count of people with repective to interest rate , we can see majority of people have interest rates between 7.5 to 15

# In[20]:


# one hot encoding for categorical variables
loan = pd.get_dummies(loan)


# In[21]:


# shuffling the data
loan =  loan.sample(frac=1)


# For data cleaning I performed one hot encoding on all the categorial feature. 
# I also checked for null values and drop the rows with null values

# ### Creating feature set

# In[22]:


# Taking the target variable 
Y = loan['interest_rate']
loan = loan.drop('interest_rate',axis = 1)


# In[23]:


# splitting the data for train and test 
x_train = loan[:8000]
x_test = loan[8001:]
y_train = Y[:8000]
y_test = Y[8001:]


# ### Linear Regression

# In[423]:


lin = LinearRegression().fit(x_train, y_train)


# As we saw that features are not highly correlated, so we are using linear regression.

# In[424]:


prid = lin.predict(x_test)
prid


# In[425]:


y_test


# In[426]:


err = mean_squared_error(y_test,prid)
print(err)


# Mean squared error , this value is  less meaning the model is predicting the values very well 

# In[427]:


plt.scatter(prid,y_test)
plt.xlabel('predicted values')
plt.ylabel('Actual values')
plt.show()


# This plot is very similar to x=y line which suggests that our predicted values are very similar to actual values 

# ### Neural Network

# In[24]:


from keras import models
from keras import layers



model = models.Sequential()
model.add(layers.Dense(units= 100,activation='linear',input_dim=x_train.shape[1]))
model.add(layers.Dense(units= 100,activation='linear'))
model.add(layers.Dense(units= 100,activation='tanh'))
model.add(layers.Dense(1, activation='relu'))

model.compile(optimizer= 'adam', loss='mean_squared_error', metrics=['mse'])


# In[25]:


modelfit=model.fit(x_train, y_train,batch_size=5, epochs=30,validation_split=0.2)


# In[27]:


evalu1 = model.evaluate(x_test, y_test,batch_size=35)
print(' Test set loss:', evalu1[0])


# In[431]:


priid = model.predict(x_test)


# In[432]:


plt.scatter(priid*2,y_test)
plt.show()


# I used simple neural network with three hidden layers and a output layer. I used relu activation function for the output layer. I used adam as the optimizer. The loss on on the test set is 24.965. 
# From a above graph we can see that neural networks didnt predict well.

# #### Final Conclusion:
# #### Linear regression beging a simple model worked well on this data. As we can see the predicted values are very close to actual value.
# ##### Neural Networks didnot perform well maybe because of less number of obeservation and many number of categorical data

# #### If I hade more time:
#     
# I can do better feature extaction using lasso regression as the features are not highly correlated.
# I would also analyze each and every numerical data's distribution and would perform appropiate scaling.

# In[ ]:




