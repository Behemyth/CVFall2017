% radial basis function for thin plate spline interpolation.
function [ result ] = phi( r )
    result = r^2*log(r);
end

