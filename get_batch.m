function y = get_batch(imdb, images, varargin)
% GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.imageSize = [473, 473] ;
opts.transformation = 'none' ;
opts.rgbMean_train = [] ;
opts.rgbMean_val = [] ;
opts.rgbMean = [] ;
opts.rgbVariance = zeros(0,3,'single') ;
%opts.classWeights = ones(1,21,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;

opts.imdbDir = [] ;
opts.modelDir = [] ;
opts.numEpochs = [] ;
opts.valEpochs = 1 ; % val in every opts.valEpoch epoch
opts.miniBatchSize = [] ;
opts.numClass = numel(imdb.objectClasses.id) ;
opts.dataset = [] ;
opts.useGpus = false ;
opts = vl_argparse(opts, varargin);

if ~isempty(images)
    switch imdb.images.set(images(1)) 
        case 1
            opts.rgbMean = opts.rgbMean_train ;
        case 2
            opts.rgbMean = opts.rgbMean_val ;
    end          
end

if ~isempty(opts.rgbVariance) && isempty(opts.rgbMean)
  opts.rgbMean = single([128;128;128]) ;
end
if ~isempty(opts.rgbMean)
  opts.rgbMean = reshape(opts.rgbMean, [1 1 3]) ;
end


% space for data
ims = [] ;
labels = [] ;
bicls_labels = [] ;
var_regul_labels = [] ;
im = cell(1,numel(images)) ;
for i=1:numel(images)
  % get data
  if isempty(im{i})
    rgbPath = sprintf(imdb.paths.image, imdb.images.name{images(i)}) ;
    labelsPath = sprintf(imdb.paths.segmentation, imdb.images.name{images(i)}) ;
    bicls_labels_path = sprintf(imdb.paths.bicls_label, imdb.images.name{images(i)}) ;
    rgb = imread(rgbPath) ;
    anno = imread(labelsPath) ;
    bicls_label = load(bicls_labels_path) ;
    bicls_label = bicls_label.bicls_label ;
  else
    rgb = im{i} ;
  end
  if size(rgb,3) == 1
    rgb = cat(3, rgb, rgb, rgb) ;
  end

  % resize image to fit model description
  im_inp = imresize(rgb, opts.imageSize); 
  % Subtract the mean (color) 
  if ~isempty(opts.rgbMean)
     ims = cat(4, ims, bsxfun(@minus, single(im_inp), opts.rgbMean)) ;
  else
     ims =cat(4, ims, im_inp) ; 
  end

  %bicls_labels
  bicls_label_inp = imresize(bicls_label, opts.imageSize, 'nearest');      
  bicls_labels = cat(4, bicls_labels, bicls_label_inp) ;  
      
  %labels
  if strcmp(opts.dataset, 'pascal')
     anno_inp = uint8(anno) ;
  else    
     anno_inp = uint8(anno) -1 ; % ignore background
  end
  anno_inp = imresize(anno_inp, opts.imageSize, 'nearest');
  labels = cat(4, labels, anno_inp) ; 
      
  %var_regul_label
  var_regul_labels = cat(4, var_regul_labels, ...
                         zeros([opts.imageSize, numel(imdb.objectClasses.id)])) ;
end

% change RGB to BGR
ims = ims(:,:,end:-1:1,:);
%transpose
ims = permute(ims, [2,1,3,4]);
labels = permute(labels, [2,1,3,4]);
bicls_labels = permute(bicls_labels, [2,1,3,4]);
var_regul_labels = permute(var_regul_labels, [2,1,3,4]);

y = {'input', ims, ...
    'bicls_labels', bicls_labels, ...
    'labels', labels, ...
    'var_regul_label', var_regul_labels} ;
