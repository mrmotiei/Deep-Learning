 # Simple RNN
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import matplotlib.pyplot as plt

t = np.arange(0,2000)
X= np.sin(0.01*t)

fig, ax = plt.subplots(figsize=(8,4))
plt.plot(X)

X_train, X_test = X[:1500],X[1500:]

model = Sequential()
model.add(SimpleRNN(units=64,activation='tanh'))
model.add(Dense(1))

model.compile(loss='mean_absolute_error',optimizer='adam')

# e.g.
#seqence = 3 steps
#[
#    [0.        , 0.00999983, 0.01999867] -> [0.0299955]
#    [0.00999983, 0.01999867, 0.0299955]  -> [0.03998933]
#    [0.01999867, 0.0299955 , 0.03998933] -> [0.04997917]
#    ...
#]


seqence = 15
XT,yT = [],[] #XT: x train , yT: y train 
for i in range(len(X_train) - seqence ):
    d = i + seqence
    XT.append(X_train[i:d,])
    yT.append(X_train[d])

Xt,yt = [],[] #XT: x test , yT: y test 
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

model.fit(XT,yT,epochs=100,batch_size=20,verbose=0)

model.evaluate(Xt, yt)

XTPredicted=model.predict(XT)
XtPredicted=model.predict(Xt)
XFinal=np.concatenate([XTPredicted,XtPredicted], axis=0) 

fig, ax = plt.subplots(figsize=(8,4))
plt.plot(X,color='red',label='Actual')
plt.plot(XFinal,color='blue', label='Predict')
plt.legend()