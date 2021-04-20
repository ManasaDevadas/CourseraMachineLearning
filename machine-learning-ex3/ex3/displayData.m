function [h, display_array] = displayData(X, example_width)
%DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested.

% Set example_width automatically if not passed in


% Explanation --- https://www.coursera.org/learn/machine-learning/discussions/weeks/4/threads/EsISnD13Eead6Qo6D3cLkQ
%https://www.coursera.org/learn/machine-learning/discussions/weeks/4/threads/LFvXVNNFEeepsQ5w0dBy1g

%Essentially the displayData code has two main functions.  It takes an X matrix (which was loaded with 100 random images by ex3.m) and reshapes each 400 pixel image into a normalized 20x20 “cell” array. Each cell is copied to a 10x10 array of cells.  It’s divided by the maximum absolute value to set up a grayscale colormap (pixel values between -1 and +1) and uses imagesc <https://www.mathworks.com/help/matlab/ref/imagesc.html > to display the square array of cells.

%example_width = sqrt(400) because its a x * x rolled out.
%width and height are calculating how much each cell is going to be. 
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2)));
end

% Gray Image
colormap(gray);

% Compute rows, cols
[m n] = size(X);
example_height = (n / example_width);

% Compute number of items to display
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

% Between images padding
pad = 1;

% Setup blank display
% -1 corresponds to black. 
%total pixels that would be displayed in a matrix format.
%20pixels per cell(height and width are same here) + one padding * no of cells to be displayed 
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));

% Copy each example into a patch on the display array
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m, 
			break; 
		end
		% Copy the patch
		
		% Get the max value of the patch
		%It finds the maximum positive or negative value in the current row of the X matrix.
		%matrix(vector,vector) select submatrix.. to get a square form is easy -- matrix([1 2 3] , [1 2 3]) selects the square matrix starting from 1 row/column to 3rd row/column
		%again loop will get matrix([1 2 3] , [4 5 6]) - next square matrix along the width/columns. 
		%repeat for next cell down the row. (j - 1) * (example_height + pad) - helps to move that many pixels along the cell
		max_val = max(abs(X(curr_ex, :)));
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val; %reshape made the row to 20 x 20 
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end
size(display_array)
% Display Image
h = imagesc(display_array, [-1 1]);

% Do not show axis
axis image off

drawnow;

end
