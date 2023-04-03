#!/usr/bin/env python
# coding: utf-8

# <img src="https://logos-world.net/wp-content/uploads/2021/08/7-Eleven-Emblem.png" width='300' height="150" >

# In[565]:



import os
import os                                        # for customising home directory
import math                
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn

from statsmodels.formula import api                # library used for model training ( better statisics)
from sklearn.linear_model import LinearRegression  # Another library used for model training 
from sklearn.feature_selection import RFE          # library used to reduce collinearity and feature selection
from sklearn.preprocessing import StandardScaler   # used for Standardasing
from sklearn.model_selection import train_test_split # used for train/test splits

from IPython.display import display # function used to render appropriate mehod to display objects # new function in week


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error # used for performance metrics

import matplotlib.pyplot as plt # used for plotting
import warnings # used to set how much warnings should be displayed
warnings.filterwarnings('ignore')


# In[567]:


plt.rcParams['figure.figsize'] = [7,4] # inch


# In[568]:


sales = pd.read_csv('seven_eleven_sales.csv')

sales.head()

original_sales=sales.copy(deep=True)

print(' The given sales data consists of {0} data entries (rows) across {1} columns. ' .format(sales.shape[0],sales.shape[1]))


# In[569]:


original_sales.shape


# In[570]:


original_sales.shape


# In[571]:


original_sales.nunique().sort_values()


# In[572]:


original_sales.info()


# In[573]:


sales.Date


# In[574]:


sales.Date=pd.to_datetime(sales.Date,dayfirst=True)

sales['weekday']=sales.Date.dt.weekday
sales['month']=sales.Date.dt.month
sales['year']=sales.Date.dt.year

sales.drop(['Date'],axis=1,inplace=True)
sales.head()


# In[575]:


sales.head()


# In[576]:


sales.weekday.value_counts()


# In[577]:


sales.weekday.unique()


# In[578]:


target='Weekly_Sales'
features= [i for i in sales.columns if i!='Weekly_Sales']
original_sales=sales.copy(deep=True)
features


# In[579]:


sales.info()


# In[580]:


sales.describe()


# In[581]:


plt.hist(sales.Weekly_Sales,bins=200);


# In[582]:


sns.distplot(sales.Weekly_Sales,bins=20,kde_kws={'color':'Navy'},hist_kws={'color':'teal'});


# In[583]:


display(sales.describe(),sales.info())


# In[584]:


sales.duplicated().sum()


# In[585]:


duplicate_count=0
rows,columns=sales.shape

sales.drop_duplicates(inplace=True)

if sales.shape==(rows,columns):
    print("No duplicates")
else:
    print('Duplicates removed /t',str(rows-sales.shape[0]))


# In[586]:


sales.nunique()


# In[587]:


features


# In[588]:


sales.Store.nunique()


# In[589]:


num_f=[]
cat_f=[]

for i in features:
    if sales[i].nunique()<=45:
        cat_f.append(i)
    else:
        num_f.append(i)


# In[590]:


display(num_f,cat_f)


# ### <Center>Univariate Analysis of Target Variable

# In[591]:


plt.figure(figsize=[8,4])
sns.distplot(sales[target],color='g',hist_kws=dict(edgecolor='black',linewidth=2.1),bins=32);


# In[592]:


k=1
for i in range(len(cat_f)):
    if sales[cat_f[i]].nunique()>8:
        plt.subplot(4,1,k)
        sns.countplot(sales[cat_f[i]],color='teal')
        k+=1
        plt.tight_layout()
l=5        
for i in range(len(cat_f)):
    if sales[cat_f[i]].nunique()<=8:
        plt.subplot(4,2,l)
        sns.countplot(sales[cat_f[i]],color='teal')
        l+=1
        plt.tight_layout()


# In[593]:


for i in range(len(cat_f)):
    if sales[cat_f[i]].nunique()<=8:
        plt.subplot(3,3,i+1)
        sns.countplot(sales[cat_f[i]],color='teal')


# In[594]:


sales['Holiday_Flag'].value_counts()


# In[595]:


for i in range(len(num_f)):
    plt.subplot(2,4,i+1)
    sns.distplot(sales[num_f[i]],hist_kws={'edgecolor':'black'},bins=15)
    plt.tight_layout();
    
for i in range(len(num_f)):
    plt.subplot(2,4,i+5)
    sns.boxplot(sales[num_f[i]])
    plt.tight_layout();    


# # <center> 'Bi-Variate Analysis of Numerical features to check for collinearity'

# In[596]:


sns.pairplot(sales[num_f]).map_upper(sns.regplot);


# In[597]:


plt.figure(figsize=[15,15])

plt.subplot(3,2,1)
grouped_sales=pd.DataFrame(sales['Weekly_Sales'].groupby(sales['Holiday_Flag']).agg(np.sum)).reset_index()
sns.barplot(data=grouped_sales,x='Holiday_Flag',y='Weekly_Sales')

plt.subplot(3,2,2)
grouped_sales=pd.DataFrame(sales['Weekly_Sales'].groupby(sales['Holiday_Flag']).agg(np.mean)).reset_index()
sns.barplot(data=grouped_sales,x='Holiday_Flag',y='Weekly_Sales');

plt.subplot(5,1,3)
grouped_sales=pd.DataFrame(sales['Weekly_Sales'].groupby(sales['month']).agg(np.sum)).reset_index()
sns.barplot(data=grouped_sales,x='month',y='Weekly_Sales');
plt.tight_layout()

plt.subplot(5,1,4)
grouped_sales=pd.DataFrame(sales['Weekly_Sales'].groupby(sales['month']).agg(np.mean)).reset_index()
sns.barplot(data=grouped_sales,x='month',y='Weekly_Sales');
plt.tight_layout()

plt.subplot(5,1,5)
grouped_sales=pd.DataFrame(sales['Weekly_Sales'].groupby(sales['Store']).agg(np.sum)).reset_index()
sns.barplot(data=grouped_sales,x='Store',y='Weekly_Sales');
plt.tight_layout()


# In[598]:


cat_f


# In[599]:


num_f


# In[600]:


for i in num_f:                                       #quartile= quantile
    quartile1 = sales[i].quantile(0.25)
    quartile3 = sales[i].quantile(0.75)
    
    iqr = quartile3-quartile1
    lower_fence = quartile1-1.5*iqr
    upper_fence = quartile3+1.5*iqr
    
    sales_new = sales[sales[i] >= lower_fence]
    sales_new = sales[sales[i] <= upper_fence]
    
display(sales_new.head())

print('Before outlier removal,the dataset has {0} rows'.format(sales_dummy.shape[0]))
print('After outlier removal,the dataset has {0} rows'.format(sales_new.shape[0]))


# In[601]:


X=sales_new.drop(['Weekly_Sales'],axis=1)
Y=sales_new.Weekly_Sales

Train_X,Test_X,Train_Y,Test_Y=train_test_split(X,Y,train_size=0.75,test_size=0.25,random_state=150)

print('Date before Train/test split->>',sales_new.shape,'\n Train dataset->> ',Train_X.shape,Train_Y.shape,'\n test dataset',Test_X.shape,Test_Y.shape)


# # <center> Standardization of training dataset

# In[602]:


std=StandardScaler()

# since we only need to standardize numerical features
Train_X_to_std=Train_X[Train_X.columns[Train_X.columns.isin(num_f)]]
Train_X_std=std.fit_transform(Train_X_to_std)
Train_X_std=pd.DataFrame(Train_X_std,columns=Train_X_to_std.columns)

Test_X_to_std=Test_X[Test_X.columns[Test_X.columns.isin(num_f)]]
Test_X_std=std.transform(Test_X_to_std)
Test_X_std=pd.DataFrame(Test_X_std,columns=Test_X_to_std.columns)


# In[603]:


display(Train_X_std.shape,Test_X_std.shape)


# # <center> we worked on our numerical data now its time to work on our categorical data

# In[604]:


Train_X_to_encode=Train_X[Train_X.columns[Train_X.columns.isin(cat_f)]].reset_index(drop=True)

Test_X_to_encode=Test_X[Test_X.columns[Test_X.columns.isin(cat_f)]].reset_index(drop=True)


# In[605]:


Test_X_to_encode


# In[614]:


categories_dummy_Train=pd.DataFrame()

for category in cat_f:
    categories_dummy_Train=pd.concat([categories_dummy_Train,
            pd.get_dummies(Train_X_to_encode[category],drop_first=False,prefix=str(category))],axis=1).reset_index(drop=True)

categories_dummy_Test=pd.DataFrame()

for category in cat_f:
    categories_dummy_Test=pd.concat([categories_dummy_Test,
            pd.get_dummies(Test_X_to_encode[category],drop_first=False,prefix=str(category))],axis=1).reset_index(drop=True)
    
categories_dummy_Test.shape


# # <center>  Now concatenating Train_X categorical data which is hot encoded to Train_X numerical data which is scaled

# In[607]:


Train_X_std=pd.concat([Train_X_std,categories_dummy_Train],axis=1)
Test_X_std=pd.concat([Test_X_std,categories_dummy_Test],axis=1)


# In[608]:


display(Train_X_std.shape,Train_Y.shape,Test_X_std.shape,Test_Y.shape)


# ## <center> Feature Engineering | Feature Selection using RFE

# In[609]:


from sklearn.feature_selection import RFE


train_r2=[]
test_r2=[]


max_features=Train_X_std.shape[1]-2

for i in range(max_features):
    
    lm=LinearRegression()
    rfe=RFE(lm,n_features_to_select=Train_X_std.shape[1]-i)
    rfe.fit(Train_X_std,Train_Y)
    
    LR=LinearRegression()
    LR.fit(Train_X_std.loc[:,rfe.support_],Train_Y)
    
    pred_train=LR.predict(Train_X_std.loc[:,rfe.support_])
    pred_test=LR.predict(Test_X_std.loc[:,rfe.support_])
    
    train_r2.append(r2_score(Train_Y,pred_train))
    test_r2.append(r2_score(Test_Y,pred_test))
    
plt.plot(train_r2,label='Train R2')
plt.plot(test_r2,label='Test R2')

plt.legend()
plt.grid()


# ### <center> As we can see , after dropping 25 variables the r_2 score begin to fall rapidly, so optimum features to drop can be assumed as 25

# In[610]:


lm=LinearRegression()
    
rfe=RFE(lm,n_features_to_select=Train_X_std.shape[1]-25)
rfe.fit(Train_X_std,Train_Y)
    
LR=LinearRegression()
LR.fit(Train_X_std.loc[:,rfe.support_],Train_Y)
    
pred_train=LR.predict(Train_X_std.loc[:,rfe.support_])
pred_test=LR.predict(Test_X_std.loc[:,rfe.support_])

display(r2_score(Train_Y,pred_train),r2_score(Test_Y,pred_test))


# In[611]:


# Storing our new reduced data in new variables for model training 

Train_X_std_rfe = Train_X_std.loc[:,rfe.support_]
Test_X_std_rfe = Test_X_std.loc[:,rfe.support_]


# ### Model Training | Multiple Linear Regression ( on reduced data after RFE )

# In[612]:


MLR = LinearRegression().fit(Train_X_std_rfe,Train_Y)


# In[613]:


print('The Coeffecient of the MLR model is ',MLR.coef_)

print('\n The Intercept of the MLR model is',MLR.intercept_)

