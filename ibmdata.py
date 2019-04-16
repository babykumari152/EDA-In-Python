#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np


# In[43]:


df=pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[44]:


df.head()


# check for null values

# In[140]:


df.isnull().sum()


# Check for features Scale Range

# In[45]:


column_name=list(df.columns)


# In[46]:


import matplotlib.pyplot as plt


# In[47]:


df['Attrition'].value_counts()


# In[48]:


#df['Attrition'].hist('Attrition')
df[df.dtypes[(df.dtypes=="float64")|(df.dtypes=="int64")]
                        .index.values].hist(figsize=[20,20])


# Handling Categorical Data

# In[49]:


p=df.dtypes[df.dtypes==object]
df.dtypes.index.values
df.dtypes.values


# In[50]:


from sklearn.preprocessing import LabelEncoder


# In[51]:


cat_col=[]
#cat_encoder=[]
encoded_df=pd.DataFrame()
for ix,val in zip(df.dtypes.index.values,df.dtypes.values):
    if val==object:
        cat_col.append(ix)
lenth=[str(i)for i in range(len(cat_col))]
print(cat_col)
print
for name,no in zip(cat_col,lenth):
    no = LabelEncoder()
    encoded_df[name]=no.fit_transform(df[name])
    #cat_encoder.append((name, no))
#print(cat_encoder)    
#for name, encs in cat_encoder:
   # df_c = encs.transform(df[name])
    #encoded_df.append(pd.DataFrame(df_c))
    #encoded_df = pd.concat(encoded_df, axis = 1, ignore_index  = True)
    #df_num = df.drop(cat_col, axis = 1)
    #y = pd.concat([df_num,encoded_df], axis = 1, ignore_index = True)
encoded_df.head()

   


# In[52]:


df.shape


# In[53]:


df=df.drop(cat_col,axis=1)
print(df.head())
#df1=pd.concate([df,encoded_df],axis=0)
for each in df.dtypes.index.values:
    encoded_df[each]=df[each]
    
    


# In[85]:


encoded_df['Attrition'].hist()


# Handle Imabalnce Class

# In[86]:


from imblearn.over_sampling import SMOTE


# In[87]:


y=encoded_df['Attrition']
x=encoded_df.drop('Attrition',axis=1)


# In[88]:


method=SMOTE(kind='regular')
x_s,y_s=method.fit_sample(x,y)


# In[89]:


label=pd.DataFrame(y_s)
label.hist()


# In[ ]:





# In[ ]:





# Spliting Train Test Data

# In[ ]:





# In[83]:


from sklearn.model_selection import train_test_split


# In[110]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=43)


# Feature Scaling

# In[123]:


from sklearn.preprocessing import StandardScaler


# In[124]:


scaler=StandardScaler()


# In[125]:


xs_train=scaler.fit_transform(x_train,y=None)
xs_test=scaler.fit_transform(x_test,y=None)


# In[18]:


encoded_df.columns


# In[19]:


#my_tab = pd.crosstab(index = df["Attrition"],columns="count")      # Name the count column
#my_tab.plot.bar()


# In[72]:


encoded_df.head()


# Dimensionality Reduction

# In[68]:


from sklearn.decomposition import PCA


# In[75]:


#!pip3 uninstall sklearn
#x_s.shape


# In[137]:


#df.index
pca=PCA(.99)
X_tr=pca.fit_transform(x_train)
x_te=pca.fit_transform(x_test)


# In[138]:


X_tr.shape
x_te.shape


# Apply Reggression Model without scaling the feature

# In[107]:


from sklearn.linear_model import LogisticRegression


# In[111]:


logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(x_train, y_train)


# In[115]:


pred=logistic_regression_model.predict(x_test)


# In[116]:


score_test = logistic_regression_model.score(x_test, y_test)
score_train=logistic_regression_model.score(x_train,y_train)


# In[117]:


print(score_test)
print(score_train)


# In[118]:


from sklearn import metrics


# In[120]:


cm=metrics.confusion_matrix(y_test,pred)


# In[121]:


print(cm)


# Apply Reggresion after Scaling the feature

# In[126]:


logistic_regression_model.fit(xs_train,y_train)


# In[128]:


preds=logistic_regression_model.predict(xs_test)


# In[129]:


score_test = logistic_regression_model.score(xs_test, y_test)
score_train=logistic_regression_model.score(xs_train,y_train)
print(score_test)
print(score_train)


# In[133]:


cm=metrics.confusion_matrix(y_test,preds)


# In[134]:


print(cm)


# apply regression on result of PCA

# In[139]:


logistic_regression_model.fit(X_tr,y_train)
pr=logistic_regression_model.predict(x_te)
score_test = logistic_regression_model.score(x_te, y_test)
score_train=logistic_regression_model.score(X_tr,y_train)
print(score_test)
print(score_train)


# In this Scenario PCA is not benifial to preprocess the data as its accuracy degrade after applying PCA

# In[ ]:




