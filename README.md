# Categorical Variables, Prediction
Based on Kaggle Competitions, https://www.kaggle.com/c/cat-in-the-dat-ii/overview/description, I have developed a formula that used to predict the cat in the data 
based on nominal data, ordinal data, and binary data. 

## Using Random Forest & PCA
In this tutorial, I am using a **Random Forest** Machine Learning Method and **Principal Component Analysis (PCA)** to analyse the information.
You may access the training data from the followings: 
- [Training Data](../master/train.csv)
- [Test Data](../master/test.csv)

## How the Code works 
First of all, we need to read all datasets, especially on the training and test data.


```
import numpy as np 
import pandas as pd

# Reading the Dataset 
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Differentiate the training and testing data in train_data
X = train_data.iloc[:,0:23].values
y = train_data.iloc[:,24].values

# Dealing with missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, 0:4])
X[:,0:4] = imputer.transform(X[:,0:4])

```
