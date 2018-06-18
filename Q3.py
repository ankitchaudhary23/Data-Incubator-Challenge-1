import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv('BSS.csv')


plt.scatter(dataset.weathersit[:20], dataset.cnt[:20])
plt.show()

plt.scatter(dataset.hr[24:48], dataset.cnt[24:48])
plt.show()


X = dataset.iloc[:, 2:16].values
Y = dataset.iloc[:, 16].values

x_train=X[:8645]
y_train=Y[:8645]

x_test=X[8645:13003]
y_test=Y[8645:13003]

x_predict=X[13004:17379]
  
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

import numpy as np

from sklearn.metrics import mean_squared_error

def mean_absolute_percentage_error(y_test, y_pred): 
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

y_Pred1=model.predict(x_predict)

mae=mean_absolute_percentage_error(y_test, y_pred)
print(mae)
