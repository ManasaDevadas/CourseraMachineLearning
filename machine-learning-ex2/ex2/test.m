a  = [1; 2; 3; 4];
b  = [5; 6; 7; 8];
deg = 6;
out1 = ones(size(a(:,1)));
for i = 1:deg
	i
	fprintf("****************\n")
    for j = 0:i
		i-j
		j
		fprintf("\n")
        out1(:, end+1) = (a.^(i-j)).*(b.^j);
    end
	fprintf("**************\n")
end   