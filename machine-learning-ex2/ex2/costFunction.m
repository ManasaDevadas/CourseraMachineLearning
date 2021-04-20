function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
% 
%note - while calculating sigmoid - multiply each X with corresponding parameters. so X(118  28) * theta(28 1) = sigmoid of each observation - (118  1) 
%Calculating gradient - multiply all X's corresponding to one theta parameter, 
%					    eg , x0 corresponding to theta0 of all observations 
%			                 so X' gives (28 118) - X' rows correspond to x0 x1 etc. 
%                       	 X'*(H - y)) = (28 118) * (118 1) = gradient (28 1) for each theta.

fprintf("size of theta, X, y and H\n")


size(theta)
size(X)



z =  X * theta;
H = sigmoid(z);

% we are using .* here for cost, we have to multiply y(i) with H(i) and log (1- y(i) with log (1 -H(i)) and take sum.
% even if we do matrix multiplication (commented out line below), we would get same result but, this one aligns with formula.
size(y)
size(H)


J =  (sum(-y .* log(H) - ((1 - y) .* log(1 - H))))/m;
%J = (sum(-y' * log(H) - ((1 - y)' * log(1 - H))))/m;
grad = (X' * (H - y))/m;




% =============================================================

end
