function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % %  - X' * (H-y) will do the sigma operation also. sum was needed in cost because of sum of all elements.
	% here its for each parameter so matrix multiplication will take care of that.
	
%note - while calculating sigmoid - multiply each X with corresponding parameters. so X(118  28) * theta(28 1) = sigmoid of each observation - (118  1) 
%Calculating gradient - multiply all X's corresponding to one theta parameter, 
%					    eg , x0 corresponding to theta0 of all observations 
%			                 so X' gives (28 118) - X' rows correspond to x0 x1 etc. 
%                       	 X'*(H - y)) = (28 118) * (118 1) = gradient (28 1) for each theta.


	H = X * theta
	theta = theta - (alpha * 1/m *  ( X' * (H - y)))


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
