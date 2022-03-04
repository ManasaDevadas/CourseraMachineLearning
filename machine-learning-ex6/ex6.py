import os
import  numpy as np
import re
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
from sklearn.svm import SVC




def plotData(X, y):

    #find the position/index of positive and negative in the data
    positiv =  y == 1
    neg = y == 0

    #Graph is between X1 on X axis and X2 on y axis
    pyplot.plot(X[positiv,0], X[positiv, 1], 'X')
    pyplot.plot(X[neg,0], X[neg, 1], 'o')
    pyplot.show()

def plot_boundary_linear(X, y, model):


    #find the position/index of positive and negative in the data
    positiv =  y == 1
    neg = y == 0
    pyplot.plot(X[positiv,0], X[positiv, 1], 'X')
    pyplot.plot(X[neg,0], X[neg, 1], 'o')

    # y = w(1)*x(1) + w(2) * x(2) + b , but y = 0 at decision, 0 = w(1)*x(1) + w(2) * x(2) + b therefore 
    # x2 = - (w(1)*x(1) + b)/w(2), plot the line!  
    w1=model.coef_[0][0]
    w2=model.coef_[0][1]
    b=model.intercept_[0]

    #https://www.machinelearningplus.com/plots/matplotlib-tutorial-complete-guide-python-plot-examples/ learn about axes
    #https://www.educba.com/matlab-gca/
    #https://stackoverflow.com/a/43811762/8291169 - how to easily draw line with slope and intercept values.
    axes = pyplot.gca()
    x1vlaues = np.array(axes.get_xlim())
    x2values = -1 * (w1*x1vlaues + b)/w2
    pyplot.plot(x1vlaues, x2values, '--')
    pyplot.show()

def gaussianKernel(x1, x2, sigma=2):
    
    #norm = (x1-x2).T.dot(x1-x2)
    normsq = np.square(x1 - x2)
    norm = np.sum(normsq)
    return(np.exp(-norm/(2*sigma**2)))

def plot_boundary(X, y, model):

    #https://www.geeksforgeeks.org/numpy-meshgrid-function/
    X0 = X[:, 0]
    X1 = X[:, 1]
    
    ## Create grid of points to plot on
    x0_min, x0_max = X0.min() - 0.04, X0.max() + 0.04
    x1_min, x1_max = X1.min() - 0.04, X1.max() + 0.04
    
    xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, 0.005), np.arange(x1_min, x1_max, 0.005))
  
    
    
    ## Evaluate model predictions
    z = model.predict(np.c_[xx0.ravel(), xx1.ravel()])
    # np.c_ is just concatenation for slice objects (which our xx0 and xx1 are).
    # ravel just flattens, but uses less memory since it flattens the original
    # array rather than creating a copy like flatten().
    
    # z is flat, so need to reshape
    z = z.reshape(xx0.shape)
    
    ## And plot
    pyplot.figure(figsize=(8,6))
    pyplot.scatter(X[y==0, 0], X[y==0, 1], c='y', marker='o')
    pyplot.scatter(X[y==1, 0], X[y==1, 1], c='k', marker='+')
    pyplot.contour(xx0, xx1, z, 1, colors='b')

    pyplot.show()

#Dataset 3 - find out the best C and gamma
#when using rbf kernel we are getting two sets of gamma and C

def dataset3Params(X, y, Xval, yval):

    Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10,30]
    gammas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10,30]
    errormat = np.zeros((8,8))

    accuracy_best = 0
    C_best = -1
    gamma_best = -1
    model_best = None

    for i in range(len(Cs)):
        C = Cs[i]
        for j in range(len(gammas)):
            gamma = gammas[j]
            print(C, gamma)
            model = SVC(kernel='rbf', gamma=gamma, C=C)
            model.fit(X, y)
            #predict for crossVal
            predictions = model.predict(Xval)
            errormat[i,j] = np.mean(predictions != yval)
    print(errormat)

    minErr = np.amin(errormat)
    indices = np.where(errormat == np.amin(errormat))
    print(minErr)
    print(indices)

    for C in Cs:
        for gamma in gammas:

            # Train SVM
            model = SVC(kernel='rbf', gamma=gamma, C=C)
            model.fit(X, y)

            # Evaluate on cross validation set
            pred = model.predict(Xval)
            accuracy = np.mean(pred == yval)

            if accuracy > accuracy_best:
                accuracy_best = accuracy
                C_best = C
                gamma_best = gamma
                model_best = model
    print(C_best)
    print(gamma_best)
    print(model_best)
    model = SVC(kernel='rbf', gamma=30, C=3)
    model.fit(X, y)
    pred = model.predict(Xval)
    accuracy = np.mean(pred == yval)
    print(pred)
    print(accuracy)
    plot_boundary(X, y, model)
    model = SVC(kernel='rbf', gamma=10, C=30)
    model.fit(X, y)
    print(pred)
    print(accuracy)
    plot_boundary(X, y, model)








    
def main() : 

  
    #data = loadmat(os.path.join('Data', 'ex6data1.mat'))
    data = loadmat('ex6data1.mat')

    #map X and y, convert y's form 2-D matrix (MATLAB format) to a numpy vector
    X, y = data['X'], data['y'][:, 0]
    #Plot the data
    plotData(X, y)

    #try different Cs
    model = SVC(kernel='linear', C=1)
    model.fit(X, y)

    #Plot the decision boundary
    plot_boundary_linear(X, y, model)

    model = SVC(kernel='linear', C=100)
    model.fit(X, y)
    plot_boundary_linear(X, y, model)

    #Gaussian Kernel
    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2

    print(gaussianKernel(x1, x2, sigma))

    #More complicated dataset
    data = loadmat('ex6data2.mat')
    print(data.keys())
    X = data['X']
    y = data['y']
    print(X[:5])
    print(y[:5])
    y = y.flatten()
    print(y)

    #plot the complicated data
    plotData(X, y)

    #Use rbf kernel
    model = SVC(kernel='rbf', gamma=50, C=1)
    model.fit(X, y)

     #plot the boundary
    plot_boundary(X, y, model)

    #Dataset 3
    data = loadmat('ex6data3.mat')
    print(data.keys())
    X = data['X']
    y = data['y']

    Xval = data['Xval']
    yval = data['yval']
    X[:5]
    y[:5]
    y = y.flatten()
    yval = yval.flatten()

    #Plot dataset3
    plotData(X, y)

    #find best C and gamma
    dataset3Params(X, y, Xval, yval)


if __name__ == "__main__":
        main()