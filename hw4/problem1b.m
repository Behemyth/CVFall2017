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

[xin, yin] = getpts;
A = [xin,yin];
% xout = zeros(size(xin), 1);
% yout = zeros(size(yin), 1);

outputOld = old;
outputNew = new;

B = arrayfun(@(a) tpsinterp(a(1),a(2),oldPts,newPts),A(:));

outputOld = insertMarker(outputOld, A);
outputNew = insertMarker(outputNew, B);

% for i = 1:size(xin)
%     [xout(i), yout(i)] = tpsinterp(xin, yin, oldPts, newPts);
%     outputOld = insertMarker(outputOld, [xin(i), yin(i)]);
%     outputNew = insertMarker(outputNew, [xout(i), yout(i)]);
% end

figure
imshow(uint8(outputOld));
imwrite(uint8(outputOld),strcat(outLoc,'1.jpg'));

figure
imshow(uint8(outputNew));
imwrite(uint8(outputNew),strcat(outLoc,'2.jpg'));
