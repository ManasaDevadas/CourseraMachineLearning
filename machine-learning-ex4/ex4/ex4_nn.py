import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.misc #Used to show matrix as an image
import matplotlib.cm as cm #Used to display images in a specific colormap
import random #To pick random images to display
import scipy.optimize as opt #fmin_cg to train neural network
import itertools
from scipy.special import expit #Vectorized sigmoid function
import sys
import math






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
     
def sigmoidGradient(z):
    return(sigmoid(z)*(1-sigmoid(z)))
    
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmda):
      
    # When comparing to Octave code note that Python uses zero-indexed arrays.
    # But because Numpy indexing does not include the right side, the code is the same anyway.
    theta1 = nn_params[0:(hidden_layer_size*(input_layer_size+1))].reshape(hidden_layer_size,(input_layer_size+1))
    theta2 = nn_params[(hidden_layer_size*(input_layer_size+1)):].reshape(num_labels,(hidden_layer_size+1))
    
    m = X.shape[0]
    
    # create y in matrix
    y_matrix = pd.get_dummies(y.ravel()).to_numpy()

    ones = np.ones((m,1))
    X = np.hstack((ones, X))
    #part - 1 feed forward
    a1 = X
    #layer 2 (Hidden)
    #print(X.shape)
    #print(theta1.shape)
    z2 = X @ theta1.T
    a2 = sigmoid(z2)
    ones = np.ones((m,1))
    a2 = np.hstack((ones, a2))
    #layer 3 output layer
    z3 = a2 @ theta2.T
    H = sigmoid(z3)
    a3 = H
   
    J = (-1/m) * np.sum(np.sum(np.multiply(y_matrix, np.log(H)) + np.multiply((1-y_matrix), np.log(1-H))))
           
    reg = (lmda/(2*m))*(np.sum(np.square(theta1[:,1:])) + np.sum(np.square(theta2[:,1:])))
    
    J = J + reg
    
    #print(type(H))
    #print(type(y_matrix))
    #print(type(theta2))
    
    #Part 2 backProp
    DELTA3 = H - y_matrix
    DELTA2 = (DELTA3 @ theta2[:, 1:] * sigmoidGradient(z2))
    
    
    theta1_grad = (1/ m) * (DELTA2.T @ a1)
    theta2_grad = (1/m) * (DELTA3.T @ a2)
    
    
    theta1_grad[:, 1:] = theta1_grad[:, 1:] + (lmda / m) * theta1[:, 1:]
    theta2_grad[:, 1:] = theta2_grad[:, 1:] + (lmda / m) * theta2[:, 1:]
    
    grad = np.concatenate([theta1_grad.ravel(), theta2_grad.ravel()])
    
  
    return J, grad

def randInitializeWeights(L_in, L_out, epsilon_init=0.12):

    # You need to return the following variables correctly 
    W = np.zeros((L_out, 1 + L_in))

    # ====================== YOUR CODE HERE ======================

    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

    # ============================================================
    return W
    
def debug_initialize_weights(fan_out, fan_in):
    w = np.zeros((fan_out, 1 + fan_in))

    w = np.sin(np.arange(w.size)).reshape(w.shape) / 10

    return w
    
def compute_numerial_gradient(cost_func, theta):
    
    numgrad = np.zeros(theta.size)
    perturb = np.zeros(theta.size)

    e = 1e-4

    for i in range(theta.size):
        perturb[i] = e
        loss1, grad1 = cost_func(theta - perturb)
        loss2, grad2 = cost_func(theta + perturb)

        numgrad[i] = (loss2 - loss1) / (2 * e)
        perturb[i] = 0
        
    return numgrad

def check_nn_gradients(lmd):

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    lmd = lmd
    
    # We generatesome 'random' test data
    theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    theta2 = debug_initialize_weights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to genete X
    X = debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + np.mod(np.arange(1, m + 1), num_labels)


    # Unroll parameters
    nn_params = np.concatenate([theta1.flatten(), theta2.flatten()])

    def cost_func(p):
        return nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)

    cost, grad = cost_func(nn_params)
    
    numgrad = compute_numerial_gradient(cost_func, nn_params)
    
    print("printing both grads")
    print(np.c_[grad, numgrad])

def predict(Theta1 , Theta2 , X):
    
    
    (m,n) = X.shape
    ones = np.ones((m,1))
    p = np.zeros(m)
    X = np.hstack((ones, X))
    #part - 1 feed forward
    a1 = X
    z2 = X @ Theta1.T
    a2 = sigmoid(z2)
    ones = np.ones((m,1))
    a2 = np.hstack((ones, a2))
    #layer 3 output layer
    z3 = a2 @ Theta2.T
    H = sigmoid(z3)
    p = np.argmax(H, axis=1)
    return p

def main() : 

    input_layer_size  = 400
    hidden_layer_size = 25
    num_labels = 10 

    mat = scipy.io.loadmat('ex4data1.mat')

    X = mat["X"]
    y = mat["y"]
    
    #flatten y 
    y=y.flatten() 
    m = len(y)

    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[0:100], :]

    # Display data
    displayData(sel)
    
    
    #add ones to X 
    #ones = np.ones((m,1))
    #X = np.hstack((ones, X)) #add the intercept
    #Load the weights
    print("Loading Saved Neural Network Parameters ...")
    mat2 = scipy.io.loadmat('ex4weights.mat')
    theta1 = mat2["Theta1"]
    theta2 = mat2["Theta2"]
    #https://stackoverflow.com/questions/30597869/what-does-np-r-do-numpy
    #https://www.javatpoint.com/numpy-ravel#:~:text=numpy.-,ravel()%20in%20Python,source%20array%20or%20input%20array.
    nn_params = np.r_[theta1.ravel(), theta2.ravel()]
    
    print('params :', nn_params.shape)
    
    cost = nnCostFunction(nn_params, 400, 25, 10, X, y, 0)[0]
    print(cost)
    
    # Regularization parameter = 1
    cost = nnCostFunction(nn_params, 400, 25, 10, X, y, 1)[0]
    print(cost)
    

    print('Initializing Neural Network Parameters ...')

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
    initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)
    
    print('Checking Backpropagation ... ')
    lmd = 0
    check_nn_gradients(lmd)
    
    input('Program paused. Press ENTER to continue')
    print('Checking Backpropagation (w/ Regularization) ...')

    lmd = 3
    check_nn_gradients(lmd)
    #https://www.datacamp.com/community/tutorials/role-underscore-python
    debug_cost, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)
    print('Cost at (fixed) debugging parameters (w/ lambda = {}): {:0.6f}\n(for lambda = 3, this value should be about 0.576051)'.format(lmd, debug_cost))

    input('Program paused. Press ENTER to continue')

    # ===================== Part 9: Training NN =====================
    # You have now implemented all the code necessary to train a neural
    # network. To train your neural network, we will now use 'opt.fmin_cg'.
    #

    print('Training Neural Network ... ')

    lmd = 1

    options= {'maxiter': 100}

    #  You should also try different values of lambda


    # Create "short hand" for the cost function to be minimized
    costFunction = lambda p: nnCostFunction(p, input_layer_size,
                                        hidden_layer_size,
                                        num_labels, X, y, lmd)

    # Now, costFunction is a function that takes in only one argument
    # (the neural network parameters)
    res = opt.minimize(costFunction,
                        initial_nn_params,
                        jac=True,
                        method='TNC',
                        options=options)

    # get the solution of the optimization
    nn_params = res.x
        
    # Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1)))


    input('Program paused. Press ENTER to continue')

    # ===================== Part 10: Visualize Weights =====================
    # You can now 'visualize' what the neural network is learning by
    # displaying the hidden units to see what features they are capturing in
    # the data

    print('Visualizing Neural Network...')

    displayData(Theta1[:, 1:])

    input('Program paused. Press ENTER to continue')

# ===================== Part 11: Implement Predict =====================
# After the training the neural network, we would like to use it to predict
# the labels. You will now implement the 'predict' function to use the
# neural network to predict the labels of the training set. This lets
# you compute the training set accuracy.

    pred = predict(Theta1, Theta2, X)
  
    print(y)
    pred = pred + 1
    print(pred)
    
    print('Training set accuracy: {}'.format(np.mean(pred == y)*100))

    input('ex4 Finished. Press ENTER to exit')
    
    
    
    

if __name__ == "__main__":
        main()
                