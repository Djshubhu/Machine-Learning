# import keras
# import tensorflow as tf
import pandas as pd
import numpy as np

dataset=pd.read_csv(r"S:\Machine Learning\Dataset\Part 8 - Deep Learning\Section 39 - Artificial Neural Networks (ANN)\Python\Churn_Modelling.csv")

X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values


def one_hot_encode(data, column):
    encoder = OneHotEncoder(categories='auto')
    encoded_data = encoder.fit_transform(data[:, column].reshape(-1, 1)).toarray()
    return np.concatenate((data[:, :column], encoded_data, data[:, column + 1:]), axis=1)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_x_1=LabelEncoder()
X[:,1]=labelencoder_x_1.fit_transform(X[:,1])
labelencoder_x_2=LabelEncoder()
X[:,2]=labelencoder_x_2.fit_transform(X[:,2])
encoded_data = one_hot_encode(X, 1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2 ,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
print(X_train.shape)

from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
classifier.add(Dense(10,activation='relu'))
classifier.add(Dense(10,input_shape=(11,),activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))
classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.fit(X_train,y_train,batch_size=10,epochs=1000)

y_pred=classifier.predict(X_test)
y_pred=(y_pred > 0.5 )
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
