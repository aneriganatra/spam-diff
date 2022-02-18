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


# In[5]:


mails


# In[6]:


from nltk.stem import WordNetLemmatizer
wnl = nltk.WordNetLemmatizer()
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[ ]:





# In[7]:


corpus = []
for i in range(0, len(mails)):
    review = re.sub('[^a-zA-Z]', ' ', mails['v2'][i])
    review = review.lower()
    review = review.split()
    
    review = [wnl.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[8]:


corpus


# In[9]:


len(corpus)


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer(analyzer='word',stop_words='english')


text_vector = vectorizer.fit_transform(corpus).toarray()


text_vector.shape


# In[11]:


mails['v1']=mails['v1'].apply(lambda x:0 if x=='ham' else 1)


# In[12]:


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(text_vector,mails['v1'],test_size=0.2,random_state=20)


(x_test)


# In[13]:


from sklearn.naive_bayes import BernoulliNB


modelB = BernoulliNB()
modelB.fit(x_train,y_train)
print(modelB.score(x_train,y_train))


y_predictedB = modelB.predict(x_test)


from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_predictedB))


# In[14]:


messages = pd.read_csv('sppam.csv')
messages


# In[15]:


messages.drop(['Unnamed: 0','label_num'],axis=1)


# In[16]:


corpus1 = []
for i in range(0, len(messages)):
    review1 = re.sub('[^a-zA-Z]', ' ', messages['text'][i])
    review1 = review1.lower()
    review1 = review1.split()
    
    review1 = [ps.stem(word) for word in review1 if not word in stopwords.words('english')]
    review1 = ' '.join(review1)
    corpus1.append(review1)
    
corpus1


# In[31]:


from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer1 = TfidfVectorizer(analyzer='word',stop_words='english')


text_vector1 = vectorizer.fit_transform(corpus1).toarray()


text_vector11=pd.DataFrame(text_vector1)
text_vector11=text_vector11.iloc[:,0:6854]


# In[33]:


y_predicted1 = modelB.predict(text_vector11)


# In[40]:



messages['label']=messages['label'].apply(lambda x:0 if x=='ham' else 1)
y_test1=messages['label']


# In[57]:


from sklearn.metrics import accuracy_score

print(accuracy_score(y_test1,y_predicted1))


# In[ ]:





# In[56]:





# In[ ]:




