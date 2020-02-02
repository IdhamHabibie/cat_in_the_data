import pandas as pd
import numpy as np

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data.head()

# Reading the data information yeah. 
X = train_data.iloc[:,0:23].values
y = train_data.iloc[:,24].values

# Dealing with missing values
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, 0:4])
X[:,0:4] = imputer.transform(X[:,0:4])

# Taking out the missing values
X = pd.DataFrame(X)
for i in range(4,23):
    temp_X = X[X[i].notnull()]
    temp_Y = y[X[i].notnull()]
    
    # Assigning numbers
    X = temp_X
    y = temp_Y

X = X.values 
# Encode the Labeling Numerical Dat
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

# Testing into the dataset test.csv
X_testing = test_data.iloc[:,0:23].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_testing[:, 0:17])
X_testing[:,0:17] = imputer.transform(X_testing[:,0:17])

# Implementing Label Encoding
for i in range(4,23):
    X_testing[:, i] = str(X_testing[:,i])
    X_testing[:, i] = labelencoder_X.fit_transform((X_testing[:, i]))


# Encode the Ordinal Encoding
from sklearn.preprocessing import OrdinalEncoder
OrdinalEncoder_X = OrdinalEncoder(categories = 'auto')
X_testing[:,17:23] = OrdinalEncoder_X.fit_transform(X_testing[:,17:23])

# Implementing PCA for this part. 
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_testing = pca.fit_transform(X_testing)
explained_variance = pca.explained_variance_ratio_
# Predicting the Test set results
hasil = classifier.predict(X_testing)

hasil = pd.DataFrame(hasil)