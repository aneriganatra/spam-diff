#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


mails=pd.read_csv('spam.csv')


# In[3]:


mails.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)


# In[4]:


import re
import nltk
nltk.download('stopwords')


# In[ ]:





# In[5]:


from nltk.stem import WordNetLemmatizer
wnl = nltk.WordNetLemmatizer()
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[ ]:





# In[6]:


corpus = []
for i in range(0, len(mails)):
    review = re.sub('[^a-zA-Z]', ' ', mails['v2'][i])
    review = review.lower()
    review = review.split()
    
    review = [wnl.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[ ]:





# In[7]:


corpus


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer(analyzer='word',stop_words='english')


text_vector = vectorizer.fit_transform(corpus).toarray()


text_vector


# In[9]:


mails['v1']=mails['v1'].apply(lambda x:0 if x=='ham' else 1)


# In[10]:


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(text_vector,mails['v1'],test_size=0.2,random_state=20)


len(x_test)


# In[11]:


from sklearn.naive_bayes import BernoulliNB


modelB = BernoulliNB()
modelB.fit(x_train,y_train)
print(modelB.score(x_train,y_train))


y_predictedB = modelB.predict(x_test)


from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_predictedB))


# In[ ]:





# In[ ]:




