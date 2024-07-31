#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd

# Load the datasets
train_df = pd.read_csv('train.csv')
gender_submission_df = pd.read_csv('gender_submission.csv')

# Display the first few rows of each dataframe
train_df.head(), gender_submission_df.head()


# In[6]:


# Check for missing values and data types
train_df.info()

# Get basic statistics of the training data
train_df.describe()

# Check for missing values and data types in the gender submission dataframe
gender_submission_df.info()
gender_submission_df.describe()


# In[7]:


# Handle missing values
# Fill missing 'Age' values with the median age
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# Fill missing 'Embarked' values with the mode
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' column due to a high number of missing values
train_df.drop(columns=['Cabin'], inplace=True)

# Convert 'Sex' and 'Embarked' columns to categorical type
train_df['Sex'] = train_df['Sex'].astype('category')
train_df['Embarked'] = train_df['Embarked'].astype('category')

# Verify changes
train_df.info()


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plot the distribution of 'Survived'
sns.countplot(x='Survived', data=train_df)
plt.title('Distribution of Survived')
plt.show()

# Plot the distribution of 'Age'
sns.histplot(train_df['Age'], kde=True)
plt.title('Distribution of Age')
plt.show()

# Explore the relationship between 'Sex' and 'Survived'
sns.countplot(x='Survived', hue='Sex', data=train_df)
plt.title('Survival by Gender')
plt.show()

# Explore the relationship between 'Pclass' and 'Survived'
sns.countplot(x='Survived', hue='Pclass', data=train_df)
plt.title('Survival by Passenger Class')
plt.show()

# Explore the relationship between 'Age' and 'Survived'
sns.boxplot(x='Survived', y='Age', data=train_df)
plt.title('Survival by Age')
plt.show()


# In[ ]:




