function [ morphedimg ] = morphfaces( wcf, wcb, im1, im2, im1pts, im2pts, t )
    [height, width, colordepth] = size(im1);
    
    uvf = zeros(height, width, 2);
    uvb = zeros(height, width, 2);
    
    % generate vector fields
    for i = 1:height
        for j = 1:width
            [uf, vf] = tpsinterp(j,i,im1pts,wcf);
            [ub, vb] = tpsinterp(j,i,im2pts,wcb);
            uvf(i,j,1) = uf - j;
            uvf(i,j,2) = vf - i;
            uvb(i,j,1) = ub - j;
            uvb(i,j,2) = vb - i;
        end
    end
    
    forwardmorph = zeros(size(im1));
    backmorph = zeros(size(im1));
    for i = 1:height
        for j = 1:width
            forwardmorph(i,j,1:3) = bilinearinterp(j + (1-t)*uvf(i,j,1), i + (1-t)*uvf(i,j,2), im2);
            backmorph(i,j,1:3) = bilinearinterp(j + t*uvb(i,j,1), i + t*uvb(i,j,2), im1);
        end
    end
    
    morphedimg = t*forwardmorph + (1-t)*backmorph;
end

