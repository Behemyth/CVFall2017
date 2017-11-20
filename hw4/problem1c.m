close all;

im1Loc = input('Old image file: ','s');
im2Loc = input('New image file: ','s');
outLoc = input('Output file name: ','s');
existingPts = input('Use existing points? ');

old = imread(im1Loc);
new = imread(im2Loc);

if (~existingPts)
    [oldPts, newPts] = cpselect(old, new, 'Wait', true);
end

figure
imshow(old);

output = zeros(size(old));
wc = tpsweights(oldPts,newPts);
for i = 1:size(old,1)
    for j = 1:size(old,2)
        [xout, yout] = tpsinterp(j,i,oldPts,wc);
        output(i,j,1:3) = bilinearinterp(xout,yout,new);
    end
end

figure
imshow(uint8(output));
imwrite(uint8(output),outLoc);