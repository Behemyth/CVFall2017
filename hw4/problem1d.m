close all;

im1Loc = input('Old image file: ','s');
im2Loc = input('New image file: ','s');
outLoc = input('Output file name: ','s');

old = imread(im1Loc);
new = imread(im2Loc);

oldGray = rgb2gray(old);
newGray = rgb2gray(new);

unmatchedOldPts = detectSURFFeatures(oldGray);
unmatchedNewPts = detectSURFFeatures(newGray);

[featuresOld, validOldPts] = extractFeatures(oldGray,unmatchedOldPts);
[featuresNew, validNewPts] = extractFeatures(newGray,unmatchedNewPts);

indexPairs = matchFeatures(featuresOld, featuresNew, 'MatchThreshold', 5, 'MaxRatio', 0.3);

oldPts = double(validOldPts(indexPairs(:,1)).Location)
newPts = double(validNewPts(indexPairs(:,2)).Location)

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