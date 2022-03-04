# %load ../../../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot

from scipy.io import loadmat
from scipy.optimize import minimize






def linearRegCostFunction(theta, X, y, lambda_, return_grad=False):
    m = y.size
    
    #print(X.shape)
    #print(X)
    #print(theta.shape)
    #print(theta)
    h = X @ theta
    squared_errors = np.power((h - y), 2)
    sumofSquaredErrors = np.sum(squared_errors)
    J = sumofSquaredErrors/(2*m);
    regularization = ((lambda_/(2*m)) * np.sum(np.square(theta[1:])));
    #regularization = (lambda_ * np.sum(np.square(theta[1:]))) / (2.0 * m)
    J = J + regularization
    
    grad = np.zeros(theta.shape)
    grad = (1/m) * X.T @ (h - y)
    grad[1:] = grad[1:] + (lambda_ / m) * theta[1:]
    
    if return_grad:
        return J, grad
    else:
        return J
    
    
#def trainLinearReg(X, y, lambda_):
   

#    m, n = X.shape

#    print("value of n")
#    print(n)
#    initial_theta = np.zeros((n, 1))
#    fargs = (X, y, lambda_)
    #Note : minimize function calls linearRegCostFunction(x0, args). So positioning of args in linearRegCostFunction is important.
#    return minimize(linearRegCostFunction,x0=initial_theta, args=fargs, method="CG", options={'disp': False, 'maxiter': 200.0})

def trainLinearReg(linearRegCostFunction, X, y, lambda_=0.0):
    """
    Trains linear regression using scipy's optimize.minimize.
    Parameters
    ----------
    X : array_like
        The dataset with shape (m x n+1). The bias term is assumed to be concatenated.
    y : array_like
        Function values at each datapoint. A vector of shape (m,).
    lambda_ : float, optional
        The regularization parameter.
    maxiter : int, optional
        Maximum number of iteration for the optimization algorithm.
    Returns
    -------
    theta : array_like
        The parameters for linear regression. This is a vector of shape (n+1,).
    """
    # Initialize Theta
    initial_theta = np.zeros(X.shape[1])

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda t: linearRegCostFunction(t, X, y, lambda_,return_grad=True)

    # Now, costFunction is a function that takes in only one argument
    options = {'maxiter': 200}

    # Minimize using scipy
    res = minimize(costFunction, initial_theta, jac=True, method='TNC', options=options)
    return res
    
def learningCurve(X, y, Xval, yval, lambda_):

    m, n = X.shape
    error_train = np.zeros((m, 1))
    error_val   = np.zeros((m, 1))
    for i in range(0, m):
        Xtrain = X[:i+1]
        ytrain = y[0:i+1]
        result = trainLinearReg(linearRegCostFunction,Xtrain, ytrain, lambda_)
        theta = result.x
        error_train[i] = linearRegCostFunction(theta, Xtrain, ytrain, 0)
        error_val[i] = linearRegCostFunction(theta, Xval, yval, 0)
    return error_train, error_val 
    
def polyFeatures(X, p):

   

    # ===================== Your Code Here =====================
    # Instructions : Given a vector X, return a matrix X_poly where the p-th
    #                column of X contains the values of X to the p-th power.
    #NOTE::::
    #We are taking X 1st column ie X[:, 0] (all rows first element if you think in numpy way) and power of it is mapped to i-th column of X_poly
   X_poly = np.zeros((X.shape[0], p))

    # ====================== YOUR CODE HERE ======================

   for i in range(p):
      X_poly[:, i] = X[:, 0]**(i+1)

    # ============================================================
   return X_poly

def featureNormalize(X_poly):
    
    mu = np.mean(X_poly,axis=0)
    X_norm = X_poly - mu
    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm = X_norm/sigma
    return X_norm, mu, sigma
    
def plotFit(min_x, max_x, mu, sigma, theta, p):

#https://stackoverflow.com/questions/12575421/convert-a-1d-array-to-a-2d-array-in-numpy
    x = np.array(np.arange(min_x - 15, max_x + 25, 0.05)).reshape(-1, 1)
    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly/sigma
    X_poly = np.column_stack((np.ones((x.shape[0],1)), X_poly))
    #plotting a graph of x and y, i.e. x and theta0 + theta1x + theta2x^2 +.... 
    pyplot.plot(x, (X_poly @ theta), '--', linewidth=2)
    
    
def validationCurve(X, y, Xval, yval):

    # Selected values of lambda (you should not change this)
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    # You need to return these variables correctly.
    error_train = np.zeros(len(lambda_vec))
    error_val = np.zeros(len(lambda_vec))

    # ====================== YOUR CODE HERE ======================

    for i in range(len(lambda_vec)):
        lambda_ = lambda_vec[i]
        result = trainLinearReg(linearRegCostFunction, X, y, lambda_=lambda_)
        theta_t = result.x
        error_train[i] = linearRegCostFunction(theta_t, X, y, 0)
        error_val[i]= linearRegCostFunction(theta_t,Xval, yval,0)
        
    # ============================================================
    return lambda_vec, error_train, error_val
    
    
def learningCurveRandomSamples(X_poly, y, X_poly_val, yval, lambda_,s):
    m, n = X_poly.shape
    error_train = np.zeros((s,1))
    error_val = np.zeros((s,1))
    k = X_poly_val.shape[0]
    
    for i in range(s):
        total_error_train = 0
        total_error_val = 0
        
        for j in range(50):
            randomNos = np.random.permutation(X_poly.shape[0])
            rand_indices = randomNos[0:i+1]
            X_poly_sel = X_poly[rand_indices]
            y_sel = y[rand_indices]
            
            result = trainLinearReg(linearRegCostFunction, X_poly_sel, y_sel, lambda_)
            theta = result.x
            
            trainerror_sel = linearRegCostFunction(theta, X_poly_sel, y_sel, 0)
            total_error_train = total_error_train + trainerror_sel
            
            X_poly_val_sel = X_poly_val[rand_indices]
            yval_sel = yval[rand_indices]
            valerror_sel = linearRegCostFunction(theta, X_poly_val_sel, yval_sel, 0)
            total_error_val = total_error_val + valerror_sel
        
        error_train[i] = total_error_train/50;
        error_val[i] = total_error_val/50

    return error_train, error_val        
            



def main() : 

    data = loadmat('ex5data1.mat')
    data.keys()
    #dict_keys(['__header__', '__version__', '__globals__', 'X', 'y', 'Xtest', 'ytest', 'Xval', 'yval'])
    
# Extract train, test, validation data from dictionary
# and also convert y's form 2-D matrix (MATLAB format) to a numpy vector
    X, y = data['X'], data['y'][:, 0]
    Xtest, ytest = data['Xtest'], data['ytest'][:, 0]
    Xval, yval = data['Xval'], data['yval'][:, 0] 
    m = len(y)

    # Plot training data
    pyplot.plot(X, y, 'rx', ms=10, linewidth=7.0)
    pyplot.xlabel('Change in water level (x)')
    pyplot.ylabel('Water flowing out of the dam (y)');
    pyplot.show(block=True)
    
    # part 1 - Cost function and gradient
    theta = np.array([1, 1])
    J = linearRegCostFunction(theta, np.concatenate([np.ones((m, 1)), X], axis=1), y, 1)
    print(J)
    J , grad = linearRegCostFunction(theta, np.concatenate([np.ones((m, 1)), X], axis=1), y, 1, return_grad=True)
    print(grad)
    
    #part 2 - train LinearRegressioncostfn with scipy minimize.
    result = trainLinearReg(linearRegCostFunction,np.concatenate([np.ones((m, 1)), X], axis=1), y, 0)
    print(result)
    theta_trained = result.x
    
    #part 3 - calculate and Plot y with trained theta. theta=result.x)
    h_trained = (np.column_stack((np.ones((m,1)), X))) @ theta_trained
    pyplot.plot(X, y, marker='x', linestyle='None')
    pyplot.ylabel('Water flowing out of the dam (y)')
    pyplot.xlabel('Change in water level (x)')
    pyplot.xticks(np.arange(-50, 50, 10.0))
    pyplot.plot(X, h_trained)
    pyplot.show()

    #part 4 - Learning Curve for Linear Regression
    lambda_ = 0
    error_train, error_val = learningCurve(np.column_stack((np.ones((m,1)), X)), y, np.column_stack((np.ones((Xval.shape[0],1)), Xval)), yval, lambda_)
    pyplot.plot(range(0, m), error_train, label="Training Error")
    pyplot.plot(range(0, m), error_val, label="Validation Error")
    pyplot.legend()
    pyplot.xlabel('Number of training examples')
    pyplot.ylabel('Error')
    pyplot.show()
    print('Training Ex\tTrain Error\tCross Validation Error\n')
    for i in  range(0, m):
        print('{:d}\t\t{:f}\t{:f}\n'.format(i+1, float(error_train[i]), float(error_val[i])))
        
   
    # part 6 - Map X onto Polynomial Features and Normalize
    p = 8

    # Map X onto Polynomial Features and Normalize
    X_poly = polyFeatures(X, p)
    X_poly, mu, sigma = featureNormalize(X_poly)
    X_poly = np.concatenate([np.ones((m, 1)), X_poly], axis=1)

    # Map X_poly_test and normalize (using mu and sigma)
    X_poly_test = polyFeatures(Xtest, p)
    X_poly_test -= mu
    X_poly_test /= sigma
    X_poly_test = np.concatenate([np.ones((ytest.size, 1)), X_poly_test], axis=1)

    # Map X_poly_val and normalize (using mu and sigma)
    X_poly_val = polyFeatures(Xval, p)
    X_poly_val -= mu
    X_poly_val /= sigma
    X_poly_val = np.concatenate([np.ones((yval.size, 1)), X_poly_val], axis=1)

    print('Normalized Training Example 1:')
    X_poly[0, :]
    
    #part 7 - Train, calculate theta and Plot the learningCurves for different Lambda
    lambda_= 1
    #lambda_=0
    #lambda_=100
    result = trainLinearReg(linearRegCostFunction, X_poly, y, lambda_)
    theta = result.x
    
    pyplot.close()
    pyplot.figure(1)
    pyplot.plot(X, y, 'rx', markersize=10, linewidth=1.5)
    plotFit(min(X), max(X), mu, sigma, theta, p)
    pyplot.xlabel('Change in water level (X)') 
    pyplot.ylabel('Water flowing out of the dam (y)')
    pyplot.title ('Polynomial Regression Fit (lambda = {:f})'.format(lambda_))
    pyplot.show()

    pyplot.figure(2)
    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_)
    p1, p2 = pyplot.plot(range(1,m+1), error_train, range(1,m+1), error_val)
    pyplot.title('Polynomial Regression Learning Curve (lambda = %f)' % lambda_)
    pyplot.xlabel('Number of training examples')
    pyplot.ylabel('Error')
    pyplot.axis([0, 13, 0, 100])
    pyplot.legend((p1, p2), ('Train', 'Cross Validation'))
    pyplot.show()

    print('Polynomial Regression (lambda = %f)\n' % lambda_)
    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m):
        print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))
    
    #part 8 - Validate best fit Lambda from a set of lambdas. It will be where cross val error is min.
    lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

    pyplot.plot(lambda_vec, error_train, '-o', lambda_vec, error_val, '-o', lw=2)
    pyplot.legend(['Train', 'Cross Validation'])
    pyplot.xlabel('lambda')
    pyplot.ylabel('Error')
    pyplot.show()

    print('lambda\t\tTrain Error\tValidation Error')
    for i in range(len(lambda_vec)):
        print(' %f\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i]))
    

    #part-9 Compute test set error with optimum lambda(test set was not used anywhere).
    min_err_val_index = np.argmin(error_val)
    lambda_optimum = lambda_vec[min_err_val_index]
    print(lambda_optimum)
    lambda_ = lambda_optimum
    result = trainLinearReg(linearRegCostFunction, X_poly, y, lambda_)
    theta = result.x
    error_train_test = linearRegCostFunction(theta, X_poly_test, ytest, 0)
    print("Test set error with optimum lambda from the set:" , error_train_test)
    
    #part-10 Learning curve for random samples. 
    k = X_poly_val.shape[0]
    m = X_poly.shape[0]
    s=min(k,m)
    lambda_ = 0.01
    error_train, error_val = learningCurveRandomSamples(X_poly, y, X_poly_val, yval, lambda_,s);
    
    pyplot.close()
    p1, p2 = pyplot.plot(range(s), error_train, range(s), error_val)
    pyplot.title('Polynomial Regression Learning Curve (lambda = {:f})'.format(lambda_))
    pyplot.legend((p1, p2), ('Train', 'Cross Validation'))
    pyplot.xlabel('Number of training examples')
    pyplot.ylabel('Error')
    pyplot.axis([0, 13, 0, 150])
    pyplot.show()

    print('Training Examples\tTrain Error\tCross Validation Error\n')
    print('Train Error\tValidation Error')
    for i in range(s):
        print(' %f\t%f' % (error_train[i], error_val[i]))
    
    

if __name__ == "__main__":
        main()
                

















