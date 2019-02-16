#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
d = pd.read_csv("good.csv")


# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np


# In[2]:


d


# In[16]:


y = d.iloc[:,1]
x = d.iloc[:,2:]


# In[17]:


x


# In[24]:



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)


# In[25]:


X_train


# In[26]:


y_train


# In[88]:


classifier = LogisticRegression(tol = 0.0000000000000000000000000000000000000000001,max_iter=1000000000)
classifier.fit(X_train, y_train)


# In[89]:


y_pred = classifier.predict(X_test)


# In[90]:


y_pred


# In[91]:


a = []
for i in y_test:
    a.append(i)


# In[92]:


length = len(a)
o = 0
s = 0
while o < length:
    s = s + np.abs(a[o] - y_pred[o])
    o = o + 1
    
print(((length-s)/(length))*100)


# In[95]:


print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))


# In[ ]:





# In[ ]:




