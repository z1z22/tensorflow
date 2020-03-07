import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpb
import os
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
for m in tf,np,keras,pd,mpb:
    print(m.__name__, m.__version__)

#1导入数据
titanic = pd.read_csv('data/titanic.csv')
# print(titanic.head(5))
x ,y= titanic[["pclass", "age", "sex"]], titanic["survived"]
x["age"].fillna(x["age"].mean(), inplace=True)
x = x.to_dict(orient="records")

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=11)

tranfer = DictVectorizer()
x_train = tranfer.fit_transform(x_train)
x_test = tranfer.transform(x_test)
# print(x_train.shape)
# print(y_train.shape)

model = keras.models.Sequential([
    keras.layers.Dense(64,activation='selu',input_shape=x_train.shape[1:]),
    keras.layers.Dense(32,activation='selu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss ='mse',
         optimizer = 'adam')

print(model.summary())

history = model.fit(x_train, y_train,epochs=10,validation_data=[x_test,y_test])

model.evaluate(x_test,y_test, verbose=1)



