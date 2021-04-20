function W = debugInitializeWeights(fan_out, fan_in)
%DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
%incoming connections and fan_out outgoing connections using a fixed
%strategy, this will help you later in debugging
%   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights 
%   of a layer with fan_in incoming connections and fan_out outgoing 
%   connections using a fix set of values
%
%   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
%   the first row of W handles the "bias" terms
%

% Set W to zeros

%https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/5MXrEUfWEem5kA6PP1nHHA

%The code comes from computeNumericalGradients.m (when you quote code you should give the source).  The purpose of computeNumericalGradients is to test your cost function and gradients implementation on a small neural network (3,5,3)(3,5,3) and training set (m=5)(m=5).

%For a test case (especially in a course) it can be helpful to use the same data each time so you don’t want to use randInitializeWeights, which would produce a different result each time.  So debugInitializeWeights was written to fill the matrices with small (\approx 0.1)(≈0.1) predictable values generated by the \sinsin function.

W = zeros(fan_out, 1 + fan_in);

% Initialize W using "sin", this ensures that W is always of the same
% values and will be useful for debugging
W = reshape(sin(1:numel(W)), size(W)) / 10;

% =========================================================================

end