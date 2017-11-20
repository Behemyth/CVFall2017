% calculates the (n+3)x2 matrix of all the weights and affine coefficients
% for solving the thin plate spline function.
function [ weightsandcoeffs ] = tpsweights( oldPts, newPts )
    botNew = zeros(3,2);
    augmentedNewPts = [newPts; botNew];
    
    bigmatrix = bigmat(oldPts);
    
    weightsandcoeffs = bigmatrix\augmentedNewPts;
end

