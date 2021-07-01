# Simple RNN - Stock Market

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import matplotlib.pyplot as plt

data = pd.read_csv('../data/AABA_2006-01-01_to_2018-01-01.csv')
data.head()

data = data[['Open']]
data.head()

plt.plot(data)

X = data['Open'].values

#[
#    [0.        , 0.00999983, 0.01999867] -> [0.0299955]
#    [0.00999983, 0.01999867, 0.0299955]  -> [0.03998933]
#    [0.01999867, 0.0299955 , 0.03998933] -> [0.04997917]
#    ...
#]

X_train, X_test = X[:2500],X[2500:]

model = Sequential()
model.add(SimpleRNN(units=64,activation='tanh'))
model.add(Dense(1))

model.compile(loss='mean_absolute_error',optimizer='adam')

seqence = 30
#XT: X train , yT: y train 
XT,yT = [],[] 
for i in range(len(X_train) - seqence ):
    d = i + seqence
    XT.append(X_train[i:d,])
    yT.append(X_train[d])

#XT: X test , yT: y test 
Xt,yt = [],[]
for i in range(len(X_test) - seqence):
    d = i + seqence
    Xt.append(X_test[i:d,])
    yt.append(X_test[d])

XT = np.array(XT)
yT = np.array(yT)
Xt = np.array(Xt)
yt = np.array(yt)

XT = np.reshape(XT,(XT.shape[0],XT.shape[1],1))
Xt = np.reshape(Xt,(Xt.shape[0],Xt.shape[1],1))

history = model.fit(XT,yT,epochs=100,batch_size=20,verbose=0)

model.evaluate(Xt, yt)

XTPredicted=model.predict(XT)
XtPredicted=model.predict(Xt)
XFinal=np.concatenate([XTPredicted,XtPredicted], axis=0)

fig, ax = plt.subplots(figsize=(8,6))
plt.plot(X,color='red',label='Actual')
plt.plot(XFinal,color='blue', label='Predict')
plt.legend()