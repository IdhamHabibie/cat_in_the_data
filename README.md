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

```
#### From the X variable, we need to identify the missing values. As we know from the EDA, it seems that the Data contains Binary values, Ordinal Values, and Categorical Values. 
```
# Dealing with missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, 0:4])
X[:,0:4] = imputer.transform(X[:,0:4])

# From Range 4 to 23, since it is the categorical values (Ordinal and Categorical). 
# I am going to take out of them 
X = pd.DataFrame(X)
for i in range(4,23):
    temp_X = X[X[i].notnull()]
    temp_Y = y[X[i].notnull()]
    
    # Assigning numbers
    X = temp_X
    y = temp_Y

X = X.values 
```
#### Encode the Ordinal and Categorical Values
```
# We need to convert the Encoding area
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

# Implementing Label Encoding
for i in range(4,17):
    X[:, i] = str(X[:,i])
    X[:, i] = labelencoder_X.fit_transform((X[:, i]))

# Encode the Ordinal Encoding
from sklearn.preprocessing import OrdinalEncoder
OrdinalEncoder_X = OrdinalEncoder(categories = 'auto')
X[:,17:23] = OrdinalEncoder_X.fit_transform(X[:,17:23])
```
#### Splitting the Dataset into the Training Set and Test Set 

```
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# Implementing PCA for this part. 

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
```
#### The confusion matrix has a good precision and accuracy.  

