# Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Preprocessing
dataset     = pd.read_csv('./Churn_Modelling.csv')
X           = dataset.iloc[:,3:13].values
Y           = dataset.iloc[:,13].values 


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X_1  = LabelEncoder()
X[:,1]          = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2  = LabelEncoder()
X[:,2]          = labelencoder_X_2.fit_transform(X[:,2])

# To avoid ordering  : convert to dummy shape
A = make_column_transformer(
    (OneHotEncoder(categories='auto'), [1]), 
    remainder="passthrough")

X=A.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X , Y , test_size=0.2 , random_state=0)


from sklearn.preprocessing import StandardScaler
sc_X        = StandardScaler()
X_train     = sc_X.fit_transform(X_train)
X_test      = sc_X.transform(X_test)



# Import Keras library 
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing ANN
classifier      = Sequential()
# Adding input layer and the first hidden layer
    # Parameter tunning
classifier.add(Dense(6 , init='uniform' , activation='relu' ,input_shape=(12,)))
# Adding Scond hidden layer
classifier.add(Dense(6 , init='uniform' , activation='relu' ))
# Adding output layer
classifier.add(Dense(1 , init='uniform' , activation='sigmoid' ))

# compiling the ANN
classifier.compile(optimizer='adam' , loss='binary_crossentropy', metrics=['accuracy'])
# Fitting ANN to Training set
classifier.fit(X_train , Y_train , batch_size=10 , nb_epoch=100)

# Predict
y_pred  = classifier.predict(X_test)
