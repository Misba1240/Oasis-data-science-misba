#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


#loading Unemployment in India dataset 
df = pd.read_csv("C:/Users/91992/Downloads/archive/Unemployment in India.csv")
df


# In[3]:


df1 = pd.read_csv("C:/Users/91992/Downloads/archive/Unemployment_Rate_upto_11_2020.csv")
df1


# In[4]:


print("Rows from start are: ")
print(df.head(6))
print("\n")
print("Rows from bottom: ")
print(df.tail(8))


# In[5]:


print("Rows from start are: ")
print(df1.head(6))
print("\n")
print("Rows from bottom: ")
print(df1.tail(8))


# In[6]:


print("Shape of the data set ",df.shape)
print("Size of the data set",df.size)
print("\n")
print("Info of the dataset \n",df.info)
print("\n")
print("Descriptive statistics of the dataset \n",df.describe)


# In[7]:


print("Shape of the data set ",df1.shape)
print("Size of the data set",df1.size)
print("\n")
print("Info of the dataset \n",df1.info)
print("\n")
print("Descriptive statistics of the dataset \n",df1.describe)


# In[8]:


print("Column names in the dataset umemployment in India: \n",df.columns)
print("\n \n")
print("Column names in the dataset umemployment rate till 2020: \n",df1.columns)


# In[9]:


print(df.isnull())
print("\n")
print(df1.isnull())


# In[10]:


print("For the dataset-Unemployment in India: ")
print(df.isnull().value_counts())
print("\n")
print("For the dataset-Unemployment rate till 2020: ")
print(df1.isnull().value_counts())


# In[11]:


sns.pairplot(df)


# In[12]:


sns.pairplot(df1)


# In[13]:


fig = plt.figure(figsize = (10, 10))
sns.histplot(x=' Estimated Unemployment Rate (%)', data=df, kde=True, hue='Area')
plt.title('Unemployment according to Area')
plt.xlabel('Unemployment Rate')
plt.show()


# In[14]:


fig = plt.figure(figsize = (5, 5))
sns.lineplot(y=' Estimated Unemployment Rate (%)', x=' Date', data=df)
plt.title('Unemployment according to Date')
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.ylabel('Unemployment Rate')
plt.show()


# In[15]:


df1.columns


# In[16]:


fig = plt.figure(figsize = (30, 15))
plt.scatter(df1[' Date'], df1[' Estimated Employed'])

plt.title('Unemployment according to Region')
plt.xlabel('Region')
plt.ylabel('Unemployment Rate')
plt.show()


# In[17]:


fig = plt.figure(figsize = (30, 15))
sns.histplot(x=' Estimated Labour Participation Rate (%)', data=df, kde=True, hue='Area')
plt.title('Labour Participation according to Area')
plt.xlabel('Labour Participation Rate')
plt.show()


# In[18]:


fig = plt.figure(figsize = (7, 7))
sns.lineplot(y=' Estimated Labour Participation Rate (%)', x=' Date', data=df)
plt.title('Labour Participation according to Date')
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.ylabel('Labour Participation Rate')
plt.show()


# In[19]:


y=df[' Estimated Unemployment Rate (%)']
x=df['Region']
plt_1 = plt.figure(figsize=(10, 10))
plt.title('Umemployment Rate', fontweight='bold' ,fontsize=20)
plt.xlabel("States",fontweight='bold',fontsize=20)
plt.ylabel("Estimated Unemployment rate",fontweight='bold',fontsize=20)
plt.xticks(rotation='vertical',fontsize=12)
sns.histplot(x, color='lavender')


# In[20]:


fig = plt.figure(figsize = (9, 9))
plt.scatter(df1['Region.1'], df1[' Estimated Labour Participation Rate (%)'])
plt.title('Labour Participation according to Region')
plt.xlabel('Region')
plt.ylabel('Labour Participation Rate')
plt.show()


# In[21]:


sns.histplot(x=' Estimated Employed', data=df, kde=True, hue='Area')
plt.title('Employment according to Area')
plt.xlabel('Employment Rate')
plt.show()


# In[22]:


fig = plt.figure(figsize = (9, 9))
sns.lineplot(y=' Estimated Employed', x=' Date', data=df)
plt.title('Employment according to Date')
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.ylabel('Employment Rate')
plt.show()


# In[23]:


fig = plt.figure(figsize = (9, 9))
plt.scatter(df1['Region.1'], df1[' Estimated Employed'])
plt.title('Employment according to Region')
plt.xlabel('Region')
plt.ylabel('Employment Rate')
plt.show()


# In[25]:


get_ipython().system('pip install plotly.express')


# In[31]:


plt.figure(figsize=(12, 12))
df1.columns= ["States","Date","Frequency",
               "Estimated Unemployment Rate","Estimated Employed",
               "Estimated Labour Participation Rate","Region",
               "longitude","latitude"]
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Employed", hue="Region", data=df1)
plt.show()


# In[32]:


plt.figure(figsize=(12, 11))
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Unemployment Rate", hue="Region", data=df1)
plt.show()


# In[33]:


import plotly.express as px
unemploment = df1[["States", "Region", "Estimated Unemployment Rate"]]
figure = px.sunburst(unemploment, path=["Region", "States"], 
                     values="Estimated Unemployment Rate", 
                     width=700, height=700, color_continuous_scale="RdY1Gn", 
                     title="Unemployment Rate in India")
figure.show()


# In[ ]:




