import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ex1data1.txt', header = None) #read from dataset
X = data.iloc[:,0] # read first column
y = data.iloc[:,1] # read second column
m = len(y) # number of training example
print(data.head()) # view first few rows of the data

plt.scatter(X, y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
print(plt.show())

#Note on np.newaxis: When you read data into X, y you will observe that X, y are rank 1 arrays. rank 1 array will have a shape of (m, ) where as rank 2 arrays will have a shape #of (m,1). When operating on arrays its good to convert rank 1 arrays to rank 2 arrays because rank 1 arrays often give unexpected results.
#To convert rank 1 to rank 2 array we use someArray[:,np.newaxis].

X = X[:,np.newaxis]
y = y[:,np.newaxis]
theta = np.zeros([2,1])
iterations = 1500
alpha = 0.01
ones = np.ones((m,1))
X = np.hstack((ones, X)) # adding the intercept term
##numpy.hstack() function is used to stack the sequence of input arrays horizontally (i.e. column wise) to make a single array.##
##Syntax : numpy.hstack(tup)

#Parameters :
#tup : [sequence of ndarrays] Tuple containing arrays to be stacked. The arrays must have the same shape along all but the second axis.

#Return : [stacked ndarray] The stacked array of the input arrays.
##
##
#Computing the cost
# theta is column 
#>>> X
#array([[ 1.    ,  6.1101],
#       [ 1.    ,  5.5277],
#>>> theta
#array([[0.],
#       [0.]])
# dot is matrix multiplication. so dot(X,theta) = h for each x.
#>>> np.dot(X, theta)
#array([[0.],
#       [0.],
#       [0.],
#


def computeCost(X, y, theta):
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)
    
J = computeCost(X, y, theta)
print(J)

#Computing Gradient Descent
# gradient descent equation  - theta(j) = theta(j) - alpha/m sum[(h(x(i) - y(i))*x(j,i)] - multiply the difference with each x(i) and take sum.
# its better to do dot(X.T, temp) because else we will get a row vector, but theta is a column vector. so to subtract this is better. 
def gradientDescent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp)
        theta = theta - (alpha/m) * temp
    return theta
theta = gradientDescent(X, y, theta, alpha, iterations)
print(theta)

#compute the cost with the theta from gd
J = computeCost(X, y, theta)
print(J)

#plot the best fit line
plt.scatter(X[:,1], y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X, theta))
plt.show()

