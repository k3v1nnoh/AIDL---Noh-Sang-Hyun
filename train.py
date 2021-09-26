import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train.csv') # importing data
print(df.head()) # print first few data

# Data preprocessing
df.isnull().sum() # check for any null values - got 0
df.dtypes # check datatype

# Exploratory Data Analysis
plt.style.use('dark_background') # #checking the stream times of top 50 streamers
plt.figure(figsize = (20,7))
df['Stream time(minutes)'].head(50).plot.bar(color = 'orangered')
plt.title('Comparing the different stream times (in minutes)')
plt.xlabel('Streamers')
plt.ylabel('Count')
plt.show()


sns.countplot(x='Mature',data = df) #checking how many streams are tagged as mature
a = df[df['Mature'] == True][['Channel', 'Watch time(Minutes)', \
                              'Stream time(minutes)','Followers']].head(10) \
                              #checking the top 10 streamers with mature streams
print(a)

plt.figure(figsize=(12,8))
sns.heatmap(df[['Channel', 'Watch time(Minutes)', 'Stream time(minutes)', 'Followers',\
                'Peak viewers','Average viewers','Followers gained','Views gained',\
                'Partnered','Mature','Language']].corr(), annot = True) \
                #overall correlation between the various columns present in our data
plt.title('Overall relation between columns of the Dataset', fontsize = 20)
plt.show()

#check stats of individual streamer
def streamer(x):
    return df.loc[df['Channel']==x]

#check the details about a streamer that streams in a particular language
def lang(x): 
        return df[df['Language'] == x]\
               [['Channel','Followers','Partnered','Mature']].head(10)
#Comparing Followers vs Follwers gained
plt.figure(figsize=(12,8))
sns.lineplot(df['Followers'], df['Followers gained'], palette = "Set1")
plt.title('Streaming time v/s Average Viewers', fontsize = 20)
plt.show()
