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

A = zeros(6,2);
for i = 1:6
    [xin, yin] = getpts;
    A(i,1:2) = [xin, yin];
end

% xout = zeros(size(xin), 1);
% yout = zeros(size(yin), 1);

outputOld = old;
outputNew = new;

wc = tpsweights(oldPts, newPts);

B = zeros(6,2);
for i = 1:6
    [xout, yout] = tpsinterp(A(i,1),A(i,2),oldPts,wc);
    B(i,1:2) = [xout,yout];
end

outputOld = insertMarker(outputOld, A, 'o', 'Size', 5);
outputNew = insertMarker(outputNew, B, 'o', 'Size', 5);

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

[height, width, depth] = size(outputOld);
montage = [uint8(outputOld), uint8(outputNew)];
C = [A, B(:,1) + width, B(:,2)];
montage = insertShape(montage, 'Line', C, 'LineWidth', 3);
figure
imshow(montage);