function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S]= pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

% - we often need to center our data points by making our mean coincide with origin of our data space. we can obtain a centered data matrix by subtracting mean from data points.
% which is done in featurenormalize.m
%my note, we are doing a featurenormalize before in which we are subtracting the mean from X thats why 1/m X'X is giving the covariance matrix.
% S is eigen values , U eigenvector. 
%Sigma = (X' * X)./m ;

Sigma = 1.0/m .* X' * X;
[U, S, V] = svd(Sigma);


end
