# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset= pd.read_csv("dataset/Credit_Card_Applications.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values   

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))   
X=sc.fit_transform(X)

from minisom import MiniSom
som=MiniSom(x=10,y=10,input_len=15,sigma=1.0)
som.random_weights_init(X)
som.train_random(data=X,num_iteration=100)

from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers=['o','s']
colors=['r','g']
for i,x in enumerate(X):
    w=som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,markers[Y[i]],markeredgecolor=colors[Y[i]],markerfacecolor='None',markersize=10,markeredgewidth=2)
show()    
mappings=som.win_map(X)
frauds=np.concatenate((mappings[(1,8)],mappings[(2,7)]),axis=0)
frauds=sc.inverse_transform(frauds)

#creating input data for supervised learning
customers=dataset.iloc[:,1:].values
frauding=np.zeros(len(dataset))
#creating a dependent variable based on frauds dataset
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        frauding[i]=1
#Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
customers=sc.fit_transform(customers)
#Training ann
from keras.models import Sequential
from keras.layers import Dense
Classifier=Sequential()
Classifier.add(Dense(units=2,kernel_initializer='uniform',activation='relu',input_dim=15))   
Classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid')) 
Classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) 
Classifier.fit(customers,frauding,batch_size=1,epochs=3)
#batch size is small as data is less 
#predicting proabilties of each customers to cheat
y_pred=Classifier.predict(customers)
y_pred=np.concatenate((dataset.iloc[:,0:1].values,y_pred),axis=1)
y_pred=y_pred[y_pred[:,1].argsort()]


