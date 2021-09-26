import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
import matplotlib.pyplot as plt

df = pd.read_csv('test.csv') # importing data
print(df.head()) # print first few data
print('The shape of our features is:', df.shape) # size of data
df.describe() # detailed stats of data
df.dtypes


# Data preprocessing
df2 = pd.get_dummies(df['Language']) # One-hot encode the data using pandas get_dummies
df = df.join(df2) # add it to the original data
print(df.head(5))
print('The shape of our features is:', df.shape)

# Convert Data to Arrays
labels = np.array(df['Followers gained']) # label:what want to predict
df= df.drop('Followers gained', axis = 1) # remove labels from df
df = df.drop('Channel', axis = 1) # remove Channel for leaving float numbers out
df = df.drop('Language', axis = 1) # remove Language for leaving float numbers out
df_list = list(df.columns) # saving df names
df = np.array(df) # convert to numpy array

# Training and Testing Sets
train_df, test_df, train_labels, test_labels = \
          train_test_split(df, labels, test_size = 0.25, random_state = 300000)
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
rf = RandomForestRegressor(n_estimators = 1000, random_state = 300000) # 1000 decision trees
rf.fit(train_df, train_labels);

# Make Prediction on data set
predictions = rf.predict(test_df) # Use the forest's predict method on the test data
errors = abs(predictions - test_labels) # Calculate the absolute errors
print('Mean Absolute Error:', round(np.mean(errors), 2)) # Print out the mean absolute error (mae)

# Determine Performance Metrics
mape = 100 * (errors / test_labels) # Calculate mean absolute percentage error (MAPE)
accuracy = 100 - np.mean(mape) # Calculate and display accuracy
print('Accuracy:', round(accuracy, 2), '%.') # Print Accuracy

# Now apply to test csv from here

# Variable Importance
importances = list(rf.feature_importances_)
df_importances = [(df, round(importance, 2)) for df, importance in zip(df_list, importances)]
df_importances = sorted(df_importances, key = lambda x: x[1], reverse = True)
'''
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in df_importances] #prints out the importance of data
'''
# Visualizations
x_values = list(range(len(importances)))
plt.bar(x_values, importances, orientation = 'vertical')
plt.xticks(x_values, df_list, rotation='vertical')
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
plt.show()



