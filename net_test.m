function info = net_test()
% -------------------------------------------------------------------------
% Config
% -------------------------------------------------------------------------
close all; clc; clear all;
addpath(genpath('visualizationCode'));
addpath(genpath('evaluationCode'));
addpath './caffe-2017/matlab' % your caffe path

dataset = 'NYU_Depth_V2' ; % "pascal" or others
imdb_dir = './data/NYU_Depth_V2' ;
pred_dir = './data/NYU_Depth_V2/prediction' ;
gpu_id = 1 ; % your GPU
model_definition = './models/ICM_test_NYUDv2.prototxt';
model_weights = './models/ICM_NYU_Depth_v2.caffemodel';
visualize = false ;
im_size = [473, 473] ;

% -------------------------------------------------------------------------
% Get data
% -------------------------------------------------------------------------
imdb = load(fullfile(imdb_dir, 'imdb.mat')) ;
stats = load(fullfile(imdb_dir, 'imdbStats.mat')) ;
val = find(imdb.images.set == 2 & imdb.images.segmentation) ;
rgbMean_val = stats.rgbMean_val ;
numClass = numel(imdb.objectClasses.id) ;

if ~exist(pred_dir)
  mkdir(pred_dir) ;
end

% load class names
load('objectName150.mat');
% load pre-defined colors 
load('color150.mat');

% -------------------------------------------------------------------------
% Set gpu
% -------------------------------------------------------------------------
if ~isempty(gpu_id)
  caffe.set_mode_gpu();
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

% -------------------------------------------------------------------------
% Initialize model
% -------------------------------------------------------------------------
net = caffe.Net(model_definition, model_weights, 'test');

% -------------------------------------------------------------------------
% Test
% -------------------------------------------------------------------------
pixel_accuracy = zeros(1,numel(val)) ;
pixel_correct = zeros(1,numel(val)) ;
pixel_labeled = zeros(1,numel(val)) ;
area_intersection = zeros(numClass, numel(val)) ;
area_union = zeros(numClass, numel(val)) ;

for i = 1: numel(val)
    fprintf('dealing with the image %d for %d\n', i, numel(val)) ;
    % read image
    im = imread(sprintf(imdb.paths.image, imdb.images.name{val(i)})) ;
    if size(im,3) == 1
       im = cat(3, im, im, im) ;
    end
    % resize image to fit model description
    im_inp = imresize(im, im_size); 
    % subtract the mean (color) 
    if ~isempty(rgbMean_val)
      rgbMean_val = reshape(rgbMean_val, [1 1 3]) ;
      im_inp = bsxfun(@minus, single(im_inp), rgbMean_val) ; 
    end
    % change RGB to BGR
    im_inp = im_inp(:,:,end:-1:1,:);
    im_inp = permute(im_inp, [2,1,3,4]);
    
    % read label
    im_anno = imread(sprintf(imdb.paths.segmentation, imdb.images.name{val(i)}));    
    
    % set inpus
    inputs = {im_inp};
    % obtain predicted image and resize to original size
    time = tic ;
    tmp = net.forward(inputs);
    time = toc(time) ;    
    featmap = net.blobs('conv6_interp').get_data() ; 
    labelmap = featmap2labelmap(featmap, size(im_anno)) ;    
    if strcmp(dataset, 'pascal')
        im_anno = im_anno + 1 ;   % include background
        im_anno(im_anno==255) = 0 ;
        im_anno(im_anno==256) = 0 ;
    end
    
    % obtain accuracy 
    [pixel_accuracy(i), pixel_correct(i), pixel_labeled(i)] = pixelAccuracy(labelmap, im_anno) ;
    [area_intersection(:,i), area_union(:,i)] = intersectionAndUnion(labelmap, im_anno, numClass);
        
    % print
    fprintf('seg accuracy: %.3f', pixel_accuracy(i)) ;
    speed = 1 / time ;
    fprintf('||speed: %.1fHz\n', speed) ;
    
    
   % visualization
   if visualize   
     % color encoding
     if strcmp(dataset, 'pascal')
        im_anno = im_anno - 1 ;   % include background
        labelmap = labelmap - 1 ;
     end 
        rgb_anno = colorEncode(im_anno, colors);
        rgb_seg = colorEncode(labelmap, colors);  

        file_pred = fullfile(pred_dir, [imdb.images.name{val(i)}, '.png']);
        imwrite(rgb_seg, file_pred);
        filePred = fullfile(pred_dir, [imdb.images.name{val(i)}, 'Anno', '.png']);
        imwrite(rgb_anno, filePred); 
   end
end

caffe.reset_all();

% compute accuracy and mIoU
mean_pixel_accuracy = sum(pixel_correct)/sum(pixel_labeled);
IoU = sum(area_intersection,2)./sum(eps+area_union,2);
mean_IoU = mean(IoU);

fprintf('for all images ||pixelAccuracy: %.3f', mean_pixel_accuracy) ;
fprintf('  ||meanIoU: %.3f\n', mean_IoU) ;

stats.pAccuracy = pixel_accuracy ;
stats.meanPixelAccuracy = mean_pixel_accuracy ;
stats.areaIntersection = area_intersection ;
stats.areaUnion = area_union ;
stats.meanIoU = mean_IoU ;

save(fullfile(pred_dir  ,'stats.mat'), 'stats') ;




function labelmap = featmap2labelmap(featmap, size)
    featmap = permute(featmap, [2,1,3,4]) ;
    featmap = imresize(featmap, size, 'bilinear') ;
    [~, labelmap] = max(featmap,[],3) ;    