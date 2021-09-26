# Noh, Sang Hyun (20727628)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
import matplotlib.pyplot as plt
import csv
import datetime

import sys
print(sys.path)

df = pd.read_csv('train.csv') # importing data
print(df.head()) # print first few data
print('The shape of our features is:', df.shape) # size of data
df.describe() # detailed stats of data
df.dtypes


# Data preprocessing (not used)
'''
dfa = pd.get_dummies(df['Language']) # One-hot encode the data using pandas get_dummies
df = df.join(df2) # add it to the original data
print(df.head(5))
print('The shape of our features is:', df.shape)
'''

# Convert Data to Arrays
labels = np.array(df['Followers gained']) # label:what want to predict
df= df.drop('Followers gained', axis = 1) # remove labels from df
df = df.drop('Channel', axis = 1) # remove Channel for leaving float numbers out
df = df.drop('Language', axis = 1) # remove Language for leaving float numbers out
df_list = list(df.columns) # saving df names
df = np.array(df) # convert to numpy array

# Training and Testing Sets
train_df, test_df, train_labels, test_labels = \
          train_test_split(df, labels, test_size = 0.25, random_state = 42) #, random_state = 100
print('Training df Shape:', train_df.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing df Shape:', test_df.shape)
print('Testing Labels Shape:', test_labels.shape)

# Establish Baseline
baseline_preds = test_df[:, df_list.index('Followers')] # baseline predictions
baseline_errors = abs(baseline_preds - test_labels) # absolute error
print('Average baseline error: ', round(np.mean(baseline_errors), \
                                        2)) # average baseline error

# Train Model
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42) #, random_state = 100 # 1000 decision trees
rf.fit(train_df, train_labels);

# Make Prediction on data set
predictions = rf.predict(test_df) # Use the forest's predict method on the test data
print('The shape of our TEST DF is:', test_df.shape)
errors = abs(predictions - test_labels) # Calculate the absolute errors
print('Mean Absolute Error:', round(np.mean(errors), 2)) # Print out the mean absolute error (mae)

# Determine Performance Metrics
mape = 100 * (errors / test_labels) # Calculate mean absolute percentage error (MAPE)
accuracy = 100 - np.mean(mape) # Calculate and display accuracy
print('Accuracy:', round(accuracy, 2), '%.') # Print Accuracy

# making prediction in 'test.csv'
df2 = pd.read_csv('test.csv')
channel = df2['Channel'] # first line for final
df2 = df2.drop('Channel', axis = 1)
df2 = df2.drop('Language', axis = 1)
df2_list = list(df2.columns)
df2 = np.array(df2)
print('The shape of our DF2 is:', df2.shape)

result = rf.predict(df2) # second line for final

# writing the 'submission.csv'

df3 = pd.DataFrame(channel)
df3['Followers gained'] = result
print(df3)

df3.to_csv('submission3.csv', index=False)


# Variable Importance

importances = list(rf.feature_importances_)
df_importances = [(df, round(importance, 2)) for df, importance in zip(df_list, importances)]
df_importances = sorted(df_importances, key = lambda x: x[1], reverse = True)

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in df_importances] #prints out the importance of data


# Visualizations of variable importance
x_values = list(range(len(importances)))
plt.bar(x_values, importances, orientation = 'vertical')
plt.xticks(x_values, df_list, rotation = '15')
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
plt.show()

# Visualization of final data
df4 = pd.read_csv('test.csv')
plt.plot(df3.index, df4['Followers'],'b-', label = 'followers')
plt.plot(df3.index, df3['Followers gained'],'r-', label = 'followers gained: predictions')
plt.legend()
plt.show()


