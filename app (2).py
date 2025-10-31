#!/usr/bin/env python
# coding: utf-8

# # Menstrual Health and PCOD Risk Analysis

# ![social-media-03.jpg](attachment:social-media-03.jpg)

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('periods.csv')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


df.duplicated().sum()


# In[9]:


df = df.drop_duplicates()


# In[10]:


df.isnull().sum()


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


df.nunique()


# In[14]:


df['Height']


# In[15]:


df['Height'] = df['Height'].str.strip()
df['Height'] = df['Height'].str.replace(' ', "'")
df['Height'] = df['Height'].str.replace('"', '')
df['Height'] = df['Height'].apply(lambda x: f"{x[0]}'{x[2:]}" if "'" not in x else x)

def convert_to_inches(x):
    try:
        if "'" in x:
            feet, inches = map(int, x.split("'"))
            return feet * 12 + inches
        else:
            return pd.to_numeric(x, errors='coerce')
    except ValueError:
        return pd.to_numeric(x, errors='coerce')

df['Height'] = df['Height'].apply(convert_to_inches)


# In[16]:


df['Height']


# In[17]:


df['Unusual_Bleeding'] = df['Unusual_Bleeding'].str.lower()
df['Unusual_Bleeding'] = df['Unusual_Bleeding'].map({'yes': 'Yes', 'no': 'No'})


# In[18]:


df['Income'] = df['Income'].apply(lambda x: 'Low' if x == 0 else 'High')
df['Menses_score'] = df['Menses_score'].apply(lambda x: 'Low' if x <= 3 else 'High')


# In[19]:


object_columns = df.select_dtypes(include=['object']).columns
print("Object type columns:")
print(object_columns)

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
print("\nNumerical type columns:")
print(numerical_columns)


# In[20]:


def classify_features(df):
    categorical_features = []
    non_categorical_features = []
    discrete_features = []
    continuous_features = []

    for column in df.columns:
        if df[column].dtype == 'object':
            if df[column].nunique() < 10:
                categorical_features.append(column)
            else:
                non_categorical_features.append(column)
        elif df[column].dtype in ['int64', 'float64']:
            if df[column].nunique() < 10:
                discrete_features.append(column)
            else:
                continuous_features.append(column)

    return categorical_features, non_categorical_features, discrete_features, continuous_features


# In[21]:


categorical, non_categorical, discrete, continuous = classify_features(df)


# In[22]:


print("Categorical Features:", categorical)
print("Non-Categorical Features:", non_categorical)
print("Discrete Features:", discrete)
print("Continuous Features:", continuous)


# In[23]:


for i in categorical:
    print(i, ':')
    print(df[i].unique())
    print()


# In[24]:


for i in categorical:
    print(i, ':')
    print(df[i].value_counts())
    print()


# In[25]:


for i in categorical:
    plt.figure(figsize=(15, 6))
    ax = sns.countplot(x=i, data=df, palette='hls')
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.show()


# In[26]:


for i in categorical:
    plt.figure(figsize=(20,10)) 
    plt.pie(df[i].value_counts(), labels=df[i].value_counts().index, autopct='%1.1f%%', textprops={'fontsize': 15,
                                           'color': 'black',
                                           'weight': 'bold',
                                           'family': 'serif' }) 
    hfont = {'fontname':'serif', 'weight': 'bold'}
    plt.title(i, size=20, **hfont) 
    plt.show()


# In[27]:


for i in discrete:
    print(i, ':')
    print(df[i].unique())
    print()


# In[28]:


for i in discrete:
    print(i, ':')
    print(df[i].value_counts())
    print()


# In[29]:


for i in discrete:
    plt.figure(figsize=(15, 6))
    ax = sns.countplot(x=i, data=df, palette='hls')
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.show()


# In[30]:


for i in discrete:
    plt.figure(figsize=(20,10)) 
    plt.pie(df[i].value_counts(), labels=df[i].value_counts().index, autopct='%1.1f%%', textprops={'fontsize': 15,
                                           'color': 'black',
                                           'weight': 'bold',
                                           'family': 'serif' }) 
    hfont = {'fontname':'serif', 'weight': 'bold'}
    plt.title(i, size=20, **hfont) 
    plt.show()


# In[31]:


for i in continuous:
    plt.figure(figsize=(15,6))
    sns.histplot(df[i], bins = 20, kde = True, palette='hls')
    plt.show()


# In[32]:


for i in continuous:
    plt.figure(figsize=(15,6))
    sns.distplot(df[i], bins = 20, kde = True)
    plt.show()


# In[33]:


for i in continuous:
    plt.figure(figsize=(15,6))
    sns.boxplot(i, data = df, palette='hls')
    plt.show()


# In[34]:


for i in continuous:
    plt.figure(figsize=(15,6))
    sns.boxenplot(i, data = df, palette='hls')
    plt.show()


# In[35]:


for i in continuous:
    plt.figure(figsize=(15,6))
    sns.violinplot(i, data = df, palette='hls')
    plt.show()


# In[36]:


for i in continuous:
    for j in continuous:
        if i != j:
            plt.figure(figsize=(15,6))
            sns.lineplot(x = i, y = j, data = df, ci = None, palette='hls')
            plt.xticks(rotation = 90)
            plt.show()


# In[37]:


for i in continuous:
    for j in continuous:
        if i != j:
            plt.figure(figsize=(15,6))
            sns.scatterplot(x = i, y = j, data = df, ci = None, palette='hls')
            plt.xticks(rotation = 90)
            plt.show()


# In[38]:


for i in categorical:
    for j in continuous:
        plt.figure(figsize=(15,6))
        sns.boxplot(x = df[i], y = df[j], data = df, palette = 'hls')
        plt.show()


# In[39]:


for i in categorical:
    for j in continuous:
        plt.figure(figsize=(15,6))
        sns.violinplot(x = df[i], y = df[j], data = df, palette = 'hls')
        plt.show()


# In[40]:


pivot_categorical = pd.pivot_table(df, values=continuous, index=categorical, 
                                   aggfunc='mean')


# In[41]:


pivot_categorical


# In[42]:


cross_tab_discrete = pd.crosstab(index=df['Unusual_Bleeding'], 
                                 columns=df['number_of_peak'])


# In[43]:


cross_tab_discrete


# In[44]:


plt.figure(figsize=(15,6))
sns.pairplot(df[continuous])
plt.show()


# In[45]:


plt.figure(figsize=(10, 8))
sns.heatmap(df[continuous].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# # Thanks !!! 
# # Copyright@ Prof. Nirmal Gaud
