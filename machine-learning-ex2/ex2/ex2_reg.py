import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

data = pd.read_csv('ex2data2.txt', header=None, names=['Test 1', 'Test 2', 'qaPassed'])


print(data.head());
print(data.keys())
print(data.shape)


X = data.iloc[:,0:2]
y = data.iloc[:, 2]
testX = X


print(X.head())
print(y.head())
print(type(X))
print(type(y))



def plot_data(X,y):

    qaPassed = np.where(y[:,0] == 1);
    NotqaPassed = np.where(y[:,0] == 0);

    print(type(qaPassed))
    print(qaPassed)
    
    #easier when not using labels.
    #xAxisqaPassed=X[qaPassed]
    ex1qaPassed = X['Test 1'].iloc[qaPassed]
    print("type ex1qaPassed")
    print(type(ex1qaPassed))
    ex2qaPassed = X['Test 2'].iloc[qaPassed]
    plt.plot(ex1qaPassed, ex2qaPassed, 'o', ms=8, mew=2, color='blue', label='qaPassed')
    ex1NotqaPassed = X['Test 1'].iloc[NotqaPassed]
    ex2NotqaPassed = X['Test 2'].iloc[NotqaPassed]
    plt.plot(ex1NotqaPassed, ex2NotqaPassed, '+', ms=8, mew=2, color='black', label='NotqaPassed')
    plt.xlabel("Microchip Test 1") 
    plt.ylabel("Microchop Test 2")
    plt.title('Scatter plot of training data')
    plt.legend()
    plt.show()


#note In python 3.5, the @ operator was introduced for matrix multiplication , similar to np.dot 
# we have to multiply each y with corresponding H, hence multiply fn.

def mapFeature(X1, X2):
    degree = 6
    out = np.ones((X1.shape[0],1))
    for i in range(1, degree+1):
        for j in range(0, i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j),np.power(X2,j))))
            print("outvalue")
            print(out)
    return out
    

def sigmoid(z):
    return 1/(1 + np.exp(-z)) 
#Goto: https://stackoverflow.com/questions/42594695/how-to-apply-a-function-map-values-of-each-element-in-a-2d-numpy-array-matrix?rq=1

def costFunctionReg(theta, X, y, lambda_t):
    z = X @ theta
    H = sigmoid(z)
    J =  (-1/m) * np.sum(np.multiply(y, np.log(H)) + np.multiply((1-y), np.log(1-H)))
    regularization = (lambda_t/(2*m)) * np.sum(np.power((theta[1:, ]),2))
    J = J + regularization
    return J

def gradientDescentReg(theta, X, y, lambda_t):
    m = len(y)
    grad = np.zeros(theta.shape)
    grad = (1/m) * X.T @ (sigmoid(X @ theta) - y)
    grad[1:] = grad[1:] + (lambda_t / m) * theta[1:]
    return grad

#
#Note on flatten() function: Unfortunately scipy’s fmin_tnc doesn’t work well with column or row vector. It expects the parameters to be in an array format. The flatten() function reduces a column or row vector into array format.
def fminunc(X, y, theta, lambda_t):
    output = opt.fmin_tnc(func = costFunctionReg, x0 = theta.flatten(), fprime = gradientDescentReg, \
                         args = (X, y.flatten(), lambda_t))
    theta = output[0]
    print(theta)
    return theta# theta contains the optimized values
    
def mapFeatureForPlotting(X1, X2):
    degree = 6
    out = np.ones(1)
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.hstack((out, 
            np.multiply(np.power(X1, i-j), 
            np.power(X2, j))))
    return out   

def plotDecisionBoundary(X,y,opt_theta):
    SIZE = 50
    u = np.linspace(-1, 1.5)
    v = np.linspace(-1, 1.5)
    z = np.zeros((SIZE, SIZE))
    
    for i in range(SIZE):
        for j in range(SIZE):
        #cannot use mapFeature because this is for a single value. X.shape wont work.
            z[i, j] = np.dot(mapFeatureForPlotting(u[i], v[j]), opt_theta)
    mask = y.flatten() == 1
    X = data.iloc[:,:-1]
    X[mask].iloc[:, 0]
    passed = plt.scatter(X[mask].iloc[:, 0], X[mask].iloc[:, 1])
    failed = plt.scatter(X[~mask].iloc[:, 0], X[~mask].iloc[:, 1])
    z = z.T
    #important to transpose z before calling contour - https://eli.thegreenplace.net/2014/meshgrids-and-disambiguating-rows-and-columns-from-cartesian-coordinates/
    plt.contour(u,v,z,0)
    plt.xlabel('Microchip Test1')
    plt.ylabel('Microchip Test2')
    plt.legend((passed, failed), ('Passed', 'Failed'))
    plt.show()
  



y = y[:, np.newaxis]
plot_data(X,y)
plt.clf()

#cannot work on dataframes with hstack. use numpy series or convert DF to numpy as below, in mapfeature you dont have to use newaxis.
X1 = data.iloc[:,0:1].to_numpy()
X2 = data.iloc[:,1:2].to_numpy()
#X1 = X.iloc[:,0]
#X2 = X.iloc[:,1]
X = mapFeature(X1, X2)
(m, n) = X.shape
theta = np.zeros((n,1)) # intializing theta with all zeros , 28 values. 
print(theta.shape)
lambda_t = 1
J = costFunctionReg(theta, X, y, lambda_t)
print(J)
grad = gradientDescentReg(theta, X, y, lambda_t)
print(grad)
opt_theta = fminunc(X, y, theta, lambda_t)

# find the model accuracy by predicting the outcomes from our learned parameters and then comparing with the original outcomes
pred = [sigmoid(np.dot(X, opt_theta)) >= 0.5]
predperc = np.mean(pred == y.flatten()) * 100
print(predperc)

#plotDecisionBoundary
plotDecisionBoundary(X,y,opt_theta)



    



    
