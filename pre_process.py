#Data Preprocessing

#Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing dataset

dataset = pd.read_csv('preprocess.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

# Taking care of missing data
# To replace nan with Value
from sklearn.impute import SimpleImputer
imputer     = SimpleImputer(missing_values= np.nan , strategy= 'mean')
imputer     = imputer.fit(X[:,1:3])
X[:,1:3]    = imputer.transform(X[:,1:3])


# Encoding categorical data
# Convert categories to numbers
from sklearn.compose import  make_column_transformer
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X  = LabelEncoder()
X[:,0]          = labelencoder_X.fit_transform(X[:,0])
# To avoid ordering  : convert to dummy shape

labelencoder_Y = LabelEncoder()
Y              = labelencoder_Y.fit_transform(Y)

A = make_column_transformer(
    (OneHotEncoder(categories='auto'), [0]), 
    remainder="passthrough")

X=A.fit_transform(X)

# Splitting data int training and testing
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X , Y  , test_size=0.2 , random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
#from sklearn.compose import make_column_transformer
sc_X        = StandardScaler()
X_train     = sc_X.fit_transform(X_train)
X_test      = sc_X.transform(X_test)





