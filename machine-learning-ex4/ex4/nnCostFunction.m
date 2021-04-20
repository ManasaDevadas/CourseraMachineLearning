function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup yones useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%part - 1 Feedforward
%layer1

X = [ones(m, 1) X];
a1 = X;
%layer 2 (Hidden)
z2 = X * Theta1';
a2 = sigmoid(X * Theta1');
a2 = [ones(m, 1) a2];
%layer 3 output layer
z3 = a2 * Theta2';
H = sigmoid(a2 * Theta2');
a3 = H;

%foreach input you will get 25 hidden outputs. so 5000 x 25. similarly H is 5000 x 10
%size(X)
%size(Theta1)
%size(a2)
%size(Theta2)
%size(H)
%H



%create 10 dimentional y matrix


%yones = ones(5000,num_labels);

%ky = [yones(:,1).*1==y, yones(:,2).*2==y,yones(:,3).*3==y, yones(:,4).*4==y, yones(:,5).*5==y, yones(:,6).*6==y,yones(:,7).*7==y, yones(:,8).*8==y, yones(:,9).*9==y, yones(:,10).*10==y];

%equating a row vector with a column vector results in comparing each element in row with full column , therefore results in that noof elements in row vector=  no of columns in the result.
ky = (1:num_labels)==y;


J = sum( (sum(-ky .* log(H) - ((1 - ky) .* log(1 - H))))/m);

%Regularized cost function

reg = (lambda/(2*m)) *((sum(sum(Theta1(:, 2:end).^2))) + (sum(sum(Theta2(:, 2:end).^2))));

J = J + reg;



%part 2 Backpropogation algorithm.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%errors
%layer 3 error
DELTA3 = H - ky;
%layer 2 error
%DELTA3 * Theta2 will give error for theta2 for each example. and multiply with each sigmoid of z2.
%think of how delta3 was calculated for each example in vectorized way in the step before. but we dont have to calculate gradient for bias in a2. (a2 has ones added after sigmoid(z2)
DELTA2 = (DELTA3 * Theta2) .* [ones(size(z2,1),1) sigmoidGradient(z2)];
DELTA2 = DELTA2(:,2:end);
%(no error in layer 1 (input_layer)

%gradients
Theta2_grad = (1/m) * DELTA3' * a2;
Theta1_grad = (1/ m) * DELTA2' * a1;

grad = [Theta1_grad(:) ; Theta2_grad(:)];

%Calculating gradients for the regularization
Theta1_grad_reg_term = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)]; % 25 x 401
Theta2_grad_reg_term = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)]; % 10 x 26

% Unroll gradients

%size(Theta2_grad)
%size(Theta2_grad)


%Adding regularization term to earlier calculated Theta_grad
Theta1_grad = Theta1_grad + Theta1_grad_reg_term;
Theta2_grad = Theta2_grad + Theta2_grad_reg_term;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
