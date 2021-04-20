function plotDecisionBoundary(theta, X, y)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
% plotting X1 on X axis and X2 on Y axis. 
%
% Plot Data
%
%%%%%%Decision boundary is  graph between X2 and X3 when h(x) is .5 which means g(z) ot theta'*X is 0  , not y and x . so we take two points from X2 and calculate X3 for that %from the equation 
%%% 
%%%%%
%%theta1 + thetha2*X1 + theta*X3 = 0
%%
%%theta3*x3 = -theta2*x2 - theta1
%%x3 = -theta2x2 - theta1/theta3.
%%    = -1/theta3*(theta2*x2 + theta1)
%%    = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
%%My explanation%%%%%%%%%%%%%%%%%%
plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    % Here is the grid range
	%The linspace function generates linearly spaced vectors. It is similar to the colon operator ":", but gives direct control over the number of points.
	% test results (X1 and X2) are in range -1 to 1.5
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
	% mapFeature for one value of u and one value of v give (1 28) - which corresponds to all features. multiply by theta to get theta'X
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
	
    z = z'; % important to transpose z before calling contour - https://eli.thegreenplace.net/2014/meshgrids-and-disambiguating-rows-and-columns-from-cartesian-coordinates/

    % Plot z = 0 
    % Notice you need to specify the range [0, 0] - range [0,0] would result in countour plot within heights 0 & 0 which is z = 0.
	%Notice that contour interprets the z matrix as a table of f(x[i], y[j]) values, so that the x axis corresponds to row number and the y axis to column number.
    contour(u, v, z, [0, 0],'LineWidth', 2)
end
hold off

end
