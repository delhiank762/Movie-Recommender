#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np


# In[8]:


movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')


# In[9]:


ratings.drop(['timestamp'], axis=1, inplace=True)


# In[10]:


movies.head()


# In[11]:


ratings.head()


# In[12]:


def replace_name(x):
    return movies[movies['movieId']==x].title.values[0]
ratings.movieId = ratings.movieId.map(replace_name)


# In[13]:


ratings.head()


# In[14]:


M = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating')


# In[15]:


M.shape


# In[16]:


M


# In[17]:


def pearson(s1,s2):
    s1_c = s1 - s1.mean()
    s2_c = s2 - s2.mean()
    return np.sum(s1_c * s2_c) / np.sqrt(np.sum(s1_c **2) * np.sum(s2_c**2))


# In[18]:


pearson(M['\'burbs, The (1989)'], M['10 Things I Hate About You (1999)'])


# In[19]:


pearson(M['Harry Potter and the Sorcerer\'s Stone (a.k.a. Harry Potter and the Philosopher\'s Stone) (2001)'],
                                                  M['Harry Potter and the Half-Blood Prince (2009)'])


# In[20]:


pearson(M['Mission: Impossible II (2000)'],M['Children of the Corn IV: The Gathering (1996)'])


# In[22]:


def get_recs(movie_name, M, num):
    import numpy as np
    reviews=[]
    for title in M.columns:
        if title == movie_name:
            continue
        cor = pearson(M[movie_name], M[title])
        if np.isnan(cor):
            continue
        else :
            reviews.append((title,cor))
            
    reviews.sort(key=lambda tup: tup[1], reverse = True)   
    return reviews[:num]


# In[23]:


recs = get_recs('Toy Story (1995)', M ,10)


# In[24]:


recs[:10]


# In[25]:


anti_recs = get_recs('Toy Story (1995)',M,8551)
anti_recs[-10:]


# In[ ]:




