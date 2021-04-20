function [error_train, error_val] = ...
    learningCurveRandomSamples(X, y, Xval, yval, lambda,s)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(s, 1);

k = size(Xval,1);
error_val   = zeros(s, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------

fprintf('\n size of X and Xval \n');
m
k
for i = 1:s,
	total_error_train = 0;
	total_error_val = 0;
		
	for j = 1:50,
		  %randomly select i no of samples.
		   index = randsample(s,i);
		   X_sel = X(index, :);
		   y_sel = y(index, :);
		   

           [theta] = trainLinearReg(X_sel, y_sel, lambda);
           [trainerror,grad] = linearRegCostFunction(X_sel, y_sel, theta, 0);
		   total_error_train = total_error_train + trainerror;
		   
		  
		   Xval_sel = Xval(index, :);
		   yval_sel = yval(index, :);
          
           [valerror,grad] = linearRegCostFunction(Xval_sel, yval_sel, theta, 0);
		   total_error_val = total_error_val+ valerror;
		  
		end;
	error_train(i) = total_error_train/50;
	error_val(i) = total_error_val/50 ; 
end;





% -------------------------------------------------------------

% =========================================================================

end
