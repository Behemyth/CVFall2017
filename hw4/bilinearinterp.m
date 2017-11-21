function [ color ] = bilinearinterp( x, y, img)
    if (x >= size(img,2) || (y >= size(img,1)) || x < 1 || y < 1)
        color = [0,0,0];
    else
        x1 = floor(x);
        y1 = floor(y);
        
        tl = (x - x1) * (y - y1);
        tr = (x1 + 1 - x) * (y - y1);
        bl = (x - x1) * (y1 + 1 - y);
        br = (x1 + 1 - x) * (y1 + 1 - y);
        
        color = img(y1,x1,1:3) * br + img(y1+1,x1,1:3) * tr + img(y1,x1+1,1:3) * bl + img(y1+1,x1+1,1:3)*tl;
    end
end

