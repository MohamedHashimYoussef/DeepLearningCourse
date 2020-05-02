# Import pacakges
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import dataset
dataset = pd.read_csv('./Restaurant_Reviews.tsv' , delimiter='\t' , quoting=3)

# Cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus  = []
for indx in range(len(dataset)):
    review  = re.sub('[^a-zA-Z]' ,' ', dataset['Review'][indx])
    review  = review.lower()
    review  = review.split()
    ps      = PorterStemmer()
    review  = [ps.stem(i) for i in review if not i in set(stopwords.words('english'))]
    review  = ' '.join(review)
    corpus.append(review)

# Creating the Bag of words model
from sklearn.feature_extraction.text import  CountVectorizer
cv      = CountVectorizer(max_features= 1500)
X       = cv.fit_transform(corpus).toarray()
Y       = dataset.iloc[: , -1].values

# Then add any calssification model