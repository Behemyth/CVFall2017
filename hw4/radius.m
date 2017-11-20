function [ r ] = radius( x1, y1, x2, y2 )
    v1 = [x1; y1];
    v2 = [x2; y2];
    
    r = norm(v1 - v2);
end

