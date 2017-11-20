% produces a large matrix for use in calculating the thin plate line
% interpolation function. oldPts is the set of control points in the old
% image.
function [ bigmatrix ] = bigmat( oldPts )
    n = size(oldPts, 1);
    
    topLeft = zeros(n);
    
    for i = 1:n
        for j = 1:n
            if (i ~= j)
                v1 = [oldPts(i,1); oldPts(i,2)];
                v2 = [oldPts(j,1); oldPts(j,2)];
                topLeft(i,j) = phi(norm(v1-v2));
            end
        end
    end
    
    bottomRight = zeros(3);
    
    topRight = [oldPts, ones(n, 1)];
    
    bottomLeft = topRight.';
    
    bigmatrix = [topLeft, topRight; bottomLeft, bottomRight];
end

