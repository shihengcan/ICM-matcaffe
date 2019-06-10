close all; clc; clear all;
% -------------------------------------------------------------------------
% Config
% -------------------------------------------------------------------------
dataset = 'NYU_Depth_V2' ;
imdb_dir = './data/NYU_Depth_V2' ;
label_dir = './data/NYU_Depth_V2/bicls_label' ;
ignore_label = -1 ; % The value of the ignored label

% -------------------------------------------------------------------------
% Get data
% -------------------------------------------------------------------------
imdb = load(fullfile(imdb_dir, 'imdb.mat')) ;
set = find(imdb.images.segmentation) ;
numClass = numel(imdb.objectClasses.id);
imdb.paths.bicls_label = [label_dir,'/%s.mat'] ;
imdb.paths.classcomp_label = [] ;
save(fullfile(imdb_dir, 'imdb.mat'), '-struct', 'imdb') ;
if ~exist(label_dir) 
    mkdir(label_dir) ;
end

% -------------------------------------------------------------------------
% Generate
% -------------------------------------------------------------------------
for i = 1: numel(set)
    fprintf('dealing with the image %d for %d \n', i, numel(set)) ;
    
    % read label
    anno = imread(sprintf(imdb.paths.segmentation, imdb.images.name{set(i)})) ;
    if strcmp(dataset, 'pascal')
       anno = anno + 1 ; % include background
    end      
    
    % generate bicls label
    bicls_label = bsxfun(@eq, anno(:), 1:numClass) ;
    
    % generate bicls label for ignored locations
    ignore_location = sum(bicls_label,2) == 0 ;
    bicls_label = single(bicls_label) ;
    bicls_label(ignore_location,:) = ignore_label ;
    
    % reshape
    bicls_label = reshape(bicls_label, [size(anno,1), size(anno,2), numClass]) ;
    
    % save bicls label
    save(fullfile(label_dir, [imdb.images.name{set(i)}, '.mat']), 'bicls_label') ; 
end