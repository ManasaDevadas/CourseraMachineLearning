import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

data = pd.read_csv('ex2data1.txt', header=None, names=['Exam 1', 'Exam 2', 'Admitted'])


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

    admitted = np.where(y[:,0] == 1);
    notAdmitted = np.where(y[:,0] == 0);

    print(type(admitted))
    print(admitted)
    
    #easier when not using labels.
    #xAxisAdmitted=X[admitted]
    ex1Admitted = X['Exam 1'].iloc[admitted]
    print("type ex1Admitted")
    print(type(ex1Admitted))
    ex2Admitted = X['Exam 2'].iloc[admitted]
    plt.plot(ex1Admitted, ex2Admitted, 'o', ms=8, mew=2, color='blue', label='admitted')
    ex1NotAdmitted = X['Exam 1'].iloc[notAdmitted]
    ex2NotAdmitted = X['Exam 2'].iloc[notAdmitted]
    plt.plot(ex1NotAdmitted, ex2NotAdmitted, '+', ms=8, mew=2, color='black', label='notAdmitted')
    plt.xlabel("Exam 1") 
    plt.ylabel("Exam 2")
    plt.title('Scatter plot of training data')
    plt.legend()
    plt.show()

def sigmoid(z):
    return 1/(1 + np.exp(-z)) 
#Goto: https://stackoverflow.com/questions/42594695/how-to-apply-a-function-map-values-of-each-element-in-a-2d-numpy-array-matrix?rq=1

#note In python 3.5, the @ operator was introduced for matrix multiplication , similar to np.dot 
# we have to multiply each y with corresponding H, hence multiply fn.
def computeSigmoidCost(theta, X, y):
    z = X @ theta
    H = sigmoid(z)
    J = (-1/m) * np.sum( np.multiply(y, np.log(H)) + np.multiply(1-y , np.log(1-H)))
    return J

def gradient(theta, X, y):
    return ((1/m) * X.T @ (sigmoid(X @ theta) - y))
    
def fminunc(theta,X,y,computeSigmoidCost):
    #Note on flatten() function: Unfortunately scipy’s fmin_tnc doesn’t work well with column or row vector. It expects the parameters to be in an array format. The flatten() function reduces a column or row vector into array format.
    temp = opt.fmin_tnc(func = computeSigmoidCost, 
                    x0 = theta.flatten(),fprime = gradient, 
                    args = (X, y.flatten()))
#   the output of above function is a tuple whose first element contains the optimized values of theta
    theta_optimized = temp[0]
    print(theta_optimized)
    return theta_optimized

def plotDecisionBoundary(X, y, theta_optimized):
    plot_x = [np.min(X[:,1]-2), np.max(X[:,2]+2)]
    plot_y = -1/theta_optimized[2]*(theta_optimized[0] 
          + np.dot(theta_optimized[1],plot_x))
    
    
    #post stacking labels are gone. 
    admitted = np.where(y[:,0] == 1);
    notAdmitted = np.where(y[:,0] == 0);
    
    ex1Admitted = X[admitted,1]
    ex2Admitted = X[admitted,2]
    print("type ex1Admitted dec bound")
    print(type(ex1Admitted))
    
    
    ex1NotAdmitted = X[notAdmitted,1]
    ex2NotAdmitted = X[notAdmitted,2]
    
    """ex1Admitted = testX['Exam 1'].iloc[admitted]
    print("type ex1Admitted")
    print(type(ex1Admitted))
    ex2Admitted = testX['Exam 2'].iloc[admitted]
    plt.plot(ex1Admitted, ex2Admitted, '+', ms=8, mew=2, color='black', label='admitted')
    ex1NotAdmitted = testX['Exam 1'].iloc[notAdmitted]
    ex2NotAdmitted = testX['Exam 2'].iloc[notAdmitted]"""
    
    admplot = plt.plot(ex1Admitted, ex2Admitted, 'o', ms=8, mew=2, color='blue',label='admitted')
    notadmplot = plt.plot(ex1NotAdmitted, ex2NotAdmitted, '+', ms=8, mew=2, color='black', label='Notadmitted')
    dec_bound = plt.plot(plot_x, plot_y,'g')
    
    plt.xlabel("Exam 1") 
    plt.ylabel("Exam 2")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #print(by_label.values())
    #print(by_label.keys())
    plt.legend(by_label.values(), by_label.keys())
    #plt.legend()
    plt.show()



y = y[:, np.newaxis]
plot_data(X,y)

plt.clf()

print(type(X))
(m, n) = X.shape
X = np.hstack((np.ones((m,1)), X))
print(type(X))

theta = np.zeros((n+1,1)) # intializing theta with all zeros
J = computeSigmoidCost(theta, X, y)
print(J)
theta_optimized = fminunc(theta,X,y,computeSigmoidCost)
print("optmized theta")
print(theta_optimized)
plotDecisionBoundary(X, y, theta_optimized)

    



    
