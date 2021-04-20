import numpy as np
import pandas as pd
data = pd.read_csv('ex1data2.txt', sep = ',', header = None)
X = data.iloc[:,0:2] # read first two columns into X
y = data.iloc[:,2] # read the third column into y
m = len(y) # no. of training samples
data.head()

# Feature Normalization

X = (X - np.mean(X))/np.std(X)

#adding intercept term and initializing parameters.
ones = np.ones((m,1))
X = np.hstack((ones, X))
alpha = 0.01
num_iters = 400
theta = np.zeros((3,1))
y = y[:,np.newaxis]

def computeCostMulti(X, y, theta):
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)
J = computeCostMulti(X, y, theta)
print(J)


def gradientDescentMulti(X, y, theta, alpha, iterations):
    m = len(y)
    
    for _ in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp)
        theta = theta - (alpha/m) * temp
    return theta
    
theta = gradientDescentMulti(X, y, theta, alpha, num_iters)
print(theta)

J = computeCostMulti(X, y, theta)
print(J)