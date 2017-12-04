im1 = imread('onesmall.jpg');
im2 = imread('twosmall.jpg');

uv = estimate_flow_interface(im1, im2, 'classic+nl-fast')