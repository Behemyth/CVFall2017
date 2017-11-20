% maps coordinates from one image to another image using thin plate spline
% interpolation.
function [ xp, yp ] = tpsinterp( x, y, oldPts, wc )
    n = size(oldPts,1);

    w = wc(1:n,1:2).';
    a = wc(n+1:n+2,1:2).';
    b = wc(n+3,1:2).';
    
    xp = a(1,1)*x + a(1,2)*y + b(1);
    yp = a(2,1)*x + a(2,2)*y + b(2);
    
    for i = 1:n
        xi = oldPts(i,1);
        yi = oldPts(i,2);
        xp = xp + w(1,i)*phi(radius(x,y,xi,yi));
        yp = yp + w(2,i)*phi(radius(x,y,xi,yi));
    end
end

