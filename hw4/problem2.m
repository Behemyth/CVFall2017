ptsf = input('Existing points file: ', 's');

% Ask user for inputs for input/output file prefixes
inf = input('Input Name: ', 's');
outf = input('Output Name: ', 's');

% Load input images
im1 = imread(strcat(inf, '1.jpg'));
im2 = imread(strcat(inf, '2.jpg'));
im3 = imread(strcat(inf, '3.jpg'));
im4 = imread(strcat(inf, '4.jpg'));
im5 = imread(strcat(inf, '5.jpg'));

if (isempty(ptsf))
    % Select control points
    [pts1, pts2] = cpselect(im1, im2, 'Wait', true);
    [pts2, pts3] = cpselect(im2, im3, pts2, pts2, 'Wait', true);
    [pts3, pts4] = cpselect(im3, im4, pts3, pts3, 'Wait', true);
    [pts4, pts5] = cpselect(im4, im5, pts4, pts4, 'Wait', true);

    % Save control points to a file (just in case)
    save problem2pts.mat pts1 pts2 pts3 pts4 pts5
else
    % Load existing points
    load(ptsf);
end

% Calculate forward/backward weights and coefficients for every morph
wc12 = tpsweights(pts1, pts2);
wc21 = tpsweights(pts2, pts1);
wc23 = tpsweights(pts2, pts3);
wc32 = tpsweights(pts3, pts2);
wc34 = tpsweights(pts3, pts4);
wc43 = tpsweights(pts4, pts3);
wc45 = tpsweights(pts4, pts5);
wc54 = tpsweights(pts5, pts4);

fprintf('starting animating sequence...\n');
% for i = 1:8
%     t = .125 * (i - 1);
%     outimg = morphfaces(wc12, wc21, im1, im2, pts1, pts2, t);
%     imwrite(uint8(outimg), sprintf('%s%d.jpg',outf,i));
%     fprintf('Finished with frame %d\n', i);
% end

parpool(8);
parfor i = 1:8
    t = .125 * (i - 1);
    outimg = morphfaces(wc23, wc32, im2, im3, pts2, pts3, t);
    imwrite(uint8(outimg), sprintf('%s%d.jpg',outf,i+8));
    fprintf('Finished with frame %d\n', i+8);
end

parfor i = 1:8
    t = .125 * (i - 1);
    outimg = morphfaces(wc34, wc43, im3, im4, pts3, pts4, t);
    imwrite(uint8(outimg), sprintf('%s%d.jpg',outf,i+16));
    fprintf('Finished with frame %d\n', i+16);
end

parfor i = 1:8
    t = .125 * (i - 1);
    outimg = morphfaces(wc45, wc54, im4, im5, pts4, pts5, t);
    imwrite(uint8(outimg), sprintf('%s%d.jpg',outf,i+24));
    fprintf('Finished with frame %d\n', i+24);
end
fprintf('done!\n');
