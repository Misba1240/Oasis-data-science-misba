#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# In[3]:


df = pd.read_csv("C:/Users/91992/Downloads/archive (1)/spam.csv",encoding = "ISO-8859-1")


# In[4]:


print("Email spam detection dataset is: \n",df)


# In[5]:


print("Top 5 rows of the dataset are: \n",df.head())
print("\n\nBottom 5 rows of the dataset are: \n",df.tail())


# In[6]:


ps=PorterStemmer()
lemmatize=WordNetLemmatizer()
corpus=[]
for i in range(0,len(df)):
  review=re.sub('[^a-zA-Z]', ' ', df['v2'][i])
  review = review.lower()
  review = review.split()
    
  review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
  review = ' '.join(review)
  corpus.append(review)


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
y=pd.get_dummies(df['v1'])
y=y.iloc[:,1].values


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)  


# In[9]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)


# In[10]:


y_pred=spam_detect_model.predict(X_test)


# In[11]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(confusion_matrix(y_test,y_pred))


# In[12]:


print("Accuracy Score {}".format(accuracy_score(y_test,y_pred)))


# In[13]:


print("Classification report: {}".format(classification_report(y_test,y_pred)))


# In[ ]:




