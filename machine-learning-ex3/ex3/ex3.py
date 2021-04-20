import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.io
import math
from scipy.special import expit 
import sys



#https://medium.com/analytics-vidhya/a-guide-to-using-logistic-regression-for-digit-recognition-with-python-codes-86aae6da10fe


def displayData(X, example_width=None):

    plt.close()
    plt.figure()
    
    if not example_width:
        example_width = int(round(math.sqrt(np.shape(X)[1])))
    #set gray image
    plt.set_cmap("gray")
    
    m, n = np.shape(X)
    
    example_height = int((n / example_width))
   
   # Compute number of items to display
    display_rows = int(math.floor(math.sqrt(m)))
    display_cols = int(math.ceil(m / display_rows))
    
    #between images padding
    pad = 1
    
    #set up blank display. 
    display_array = -np.ones((pad + display_rows * (example_height + pad),  pad + display_cols * (example_width + pad)))
    
    curr_ex = 1
    for j in range(1,display_rows+1):
        for i in range (1,display_cols+1):
            if curr_ex > m:
                exit
    
            max_val = max(abs(X[curr_ex-1, :]))
            #range is from 0, that nullifies effect of array index starting 1 in octave and 0 in python
            #initially rows (i=1 and j=1) the statement would between
            #>>> rows
            #array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
            # display_array[rows[0]:rows[-1]+1 , cols[0]:cols[-1]+1] would be
            #display_array[1:(20+1), 1:(20+1)] = display_array[1:21, 1:21]
            #which will help to get that sqaure subset of a cell
            
            rows = pad + (j - 1) * (example_height + pad) + np.array(range(example_height))
            cols = pad + (i - 1) * (example_width + pad) + np.array(range(example_width))
            #rows[0] = number , rows 
            # Order F helps in unrolling to column wise 
            display_array[rows[0]:rows[-1]+1 , cols[0]:cols[-1]+1] = np.reshape(X[curr_ex-1, :], (example_height, example_width), order="F") / max_val
            
            curr_ex += 1
            
    h = plt.imshow(display_array, vmin=-1, vmax=1)
            # Do not show axis
    plt.axis('off')

    plt.show(block=True)

    return h, display_array

def sigmoid(z):
     g = expit(z)
     return g
     

def lrCostFunction(theta, X, y, lambda_reg):
    
    m = len(y)
    # You need to return the following variables correctly 
    J = 0
    
    #print("------shape of X y and theta----------")
    #print(X.shape)
    #print(y.shape)
    #print(theta.shape)   
    
    z = X @ theta
    H = sigmoid(z)
 #   print("------shape of H , y , grad , theta----------")
 #   print(H.shape)
 #   print(y.shape)
 #   print(grad.shape)
 #   print(theta.shape)
     
     
    J =  (-1/m) * np.sum(np.multiply(y, np.log(H)) + np.multiply((1-y), np.log(1-H)))
    regularization = (lambda_reg/(2*m)) * np.sum(np.power((theta[1:, ]),2))
    J = J + regularization
   

    return J
        
def gradRegularization(theta, X, y, lambda_reg):
    
    grad = np.zeros(theta.shape)
    m = len(y)
    grad = (1/m) * X.T @ (sigmoid(X @ theta) - y)
    grad[1:] = grad[1:] + (lambda_reg / m) * theta[1:]
    
    return grad

def oneVsAll(X, y, num_labels, lambda_reg):
    # Some useful variables
    m, n = X.shape
    all_theta = np.zeros((num_labels, n))
    
    for c in range(num_labels): 
    #for c in range(1, num_labels+1): # if we are doing this, we 'd be creating theta for 1 - 9 and then for 0 means theta[0] is for number 1.
            #c if c -- will be true only if c is non-zero, 
        digit_class = c if c else 10 
        all_theta[c] = opt.fmin_cg(f = lrCostFunction, x0 = all_theta[c],  fprime = gradRegularization, args = (X, (y == digit_class), lambda_reg), maxiter = 50)
        #all_theta[c-1] = opt.fmin_cg(f = lrCostFunction, x0 = all_theta[c-1],  fprime = gradRegularization, args = (X, (y == c), lambda_reg), maxiter = 50)
        
    
    return all_theta
    
def predictOnevsAll(all_theta, X):
    num_labels = all_theta.shape[0]
    p = np.zeros((m, 1))
    #p = (np.argmax(sigmoid( np.dot(X,all_theta.T) ), axis=1) + 1)  use when using c in range(1, num_labels+1) because theta[0] is for 1 
    
    p = np.argmax(sigmoid( np.dot(X,all_theta.T) ), axis=1)
    return p
        


## Setup the parameters you will use for this part of the exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

mat = scipy.io.loadmat('ex3data1.mat')

X = mat["X"]
y = mat["y"] 




#looks like crucial for performance.
y=y.flatten() 
m = len(y)
rand_indices = np.random.permutation(m)
sel = X[rand_indices[0:100], :]

# Display data
displayData(sel)

#test data for regularized logistic regression
theta_t = np.array([[-2], [-1], [1], [2]])
arr = np.arange(start=1, stop=16, step=1)
arr1 = np.reshape(arr, (5, 3), order="F")/10
X_t = np.concatenate((np.ones((5,1)), arr1), axis=1)
y_t = np.array([[1],[0],[1],[0],[1]]) 
lambda_t = 3;



J= lrCostFunction(theta_t, X_t, y_t, lambda_t)
grad = gradRegularization(theta_t, X_t, y_t, lambda_t)

print("cost\n------------\n", J)
print("gradient\n------------\n", grad)
print('Expected cost: 2.534819')
print('Expected gradients:')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')
print(' shape\n---')
print(grad.shape)

#test ends here.

ones = np.ones((m,1))
X = np.hstack((ones, X)) #add the intercept
(m,n) = X.shape
lambda_reg = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_reg)

print(all_theta)

input('Program paused. Press enter to continue.\n')

prediction = predictOnevsAll(all_theta , X)

print(prediction)
print(y)
print(y%10)

prediction = [e if e else 10 for e in prediction] #not needed when using for c in range(1, num_labels+1):  in onevsAll function because 0 will be 10 itself.
print('Training Set Accuracy: {:f}'.format((np.mean(prediction == y)*100)))


