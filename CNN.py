# Importing Keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import  Dense

#Initializing the CNN
classifier      = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32 , 3 , 3 , input_shape=(64 , 64 , 3) , activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Falttining
classifier.add(Flatten())

# Step 4 - Full Connected layers
classifier.add(Dense(128 , activation='relu'))
classifier.add(Dense(128 , activation='sigmoid'))

# Compiling CNN
classifier.compile(optimizer='adam' , loss='binary_crossentropy', metrics=['accuracy'])

