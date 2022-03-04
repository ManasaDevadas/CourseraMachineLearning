function visualizeBoundaryLinear(X, y, model)
%VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
%SVM
%   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
%   learned by the SVM and overlays the data on it
% y = w(1)*x(1) + w(2) * x(2) + b , but y = 0 at decision, 0 = w(1)*x(1) + w(2) * x(2) + b therefore 
% x2 = - (w(1)*x(1) + b)/w(2), plot the graph!
%https://www.coursera.org/learn/machine-learning/discussions/forums/Ht6z-LQQEeuIygqL3YCl7Q/threads/HV-Jf4o2EeyLxAroUu5PRw
% 
w = model.w;
b = model.b;
xp = linspace(min(X(:,1)), max(X(:,1)), 100);
yp = - (w(1)*xp + b)/w(2);
w
b
w(1)
w(2)
fprintf("plotting X and y");
plotData(X, y);
hold on;
fprintf("plotting xp yp");
plot(xp, yp, '-b'); 
hold off

end
