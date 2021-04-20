function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

%fprintf("size of theta and X\n")
%size(theta)
%size(X)

%note - while calculating sigmoid - multiply each X with corresponding parameters. so X(118  28) * theta(28 1) = sigmoid of each observation - (118  1) 
%Calculating gradient - multiply all X's corresponding to one theta parameter, 
%					    eg , x0 corresponding to theta0 of all observations 
%			                 so X' gives (28 118) - X' rows correspond to x00 x01 etc. 
%                       	 X'*(H - y)) = (28 118) * (118 1) = gradient (28 1) for each theta.


z =  X * theta;
H = sigmoid(z);


J =  1/m * sum(-y .* log(H) - ((1 - y) .* log(1 - H)));
regularization = (lambda/(2*m)) * sum(theta(2:end).^2);
J = J + regularization;

%grad = (X'*(H - y))/m;
%% no regularization for theta0

%grad(1) = X'(1,:) * (H - y ) == > (1 118) * (118 1)
%grad(2:end) = (1/m)*(X'(2:end, :))*(H - y) == > (27 118) * (118 1)



grad(1) = (1/m)*(X'(1,:))*(H - y);

grad(2:end) = (1/m)*(X'(2:end, :))*(H - y) + (lambda/m) * theta(2:end);



% =============================================================

end
