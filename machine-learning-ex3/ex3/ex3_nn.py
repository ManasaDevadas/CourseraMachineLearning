import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.io
import math
from scipy.special import expit 
import sys






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


def predict(Theta1 , Theta2 , X):
    
    (m,n) = X.shape
    num_labels = Theta2.shape[0]
    z1 = X @ Theta1.T
    a1 = sigmoid(z1)
    ones = np.ones((m,1))
    a1 = np.hstack((ones, a1))
    a2 = a1 @ Theta2.T
    H = sigmoid(a2)
    p = np.argmax(H, axis=1)
    return p

def main() : 

    input_layer_size  = 400
    hidden_layer_size = 25
    num_labels = 10 

    mat = scipy.io.loadmat('ex3data1.mat')

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
    ones = np.ones((m,1))
    X = np.hstack((ones, X)) #add the intercept
    #Load the weights
    print("Loading Saved Neural Network Parameters ...")
    mat2 = scipy.io.loadmat('ex3weights.mat')
    Theta1 = mat2["Theta1"]
    Theta2 = mat2["Theta2"]
    
    #Predict using the weights and NN
    pred = predict(Theta1, Theta2, X)
    print(pred)
    print(y)
    pred = pred +1
    print('Training Set Accuracy: {:f}'.format((np.mean(pred == y)*100)))

if __name__ == "__main__":
        main()
                