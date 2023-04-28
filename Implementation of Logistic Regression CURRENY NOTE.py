import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as LA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
data = pd.read_csv('currency.csv')
print(data.shape)
print(data)
x1 = data['variance'].values
x2 = data['skewness'].values
x3 = data['kurtosis'].values
x4 = data['entropy'].values
Y = data['class'].values
m = len(x1)
x1 = x1.reshape(m)
x2 = x1.reshape(m)
x3 = x1.reshape(m)
x4 = x1.reshape(m)
x0=np.ones(m)
X = np.array([x0,x1,x1*2,x2,x2*2,x3,x3*2,x4,x4*2]).T ############################linear
v=3.62160
s=8.66610
k=-2.8073
e=-0.44699
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
reg=LogisticRegression()
reg.fit(X_train,y_train)
h_theta =reg.predict(X_test)
print(confusion_matrix(y_test,h_theta))
print("\nScore:",reg.score(X_train,y_train))
print("\nError_Percentage:",(42*100)/275,'%')
