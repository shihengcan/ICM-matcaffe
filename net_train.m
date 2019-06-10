function info = net_train(varargin)
% -------------------------------------------------------------------------
% Config
% -------------------------------------------------------------------------
addpath(genpath('evaluationCode'));
addpath '/home/amax/ShiHengcan/caffe-2017/matlab' % your caffe path

opts.dataset = 'NYU_Depth_V2' ; % "pascal" or others
opts.imdbDir = './data/NYU_Depth_V2' ;
opts.modelDir = './models' ;
model_weights = fullfile(opts.modelDir,'pspnet50_init.caffemodel') ;
model_solver = fullfile(opts.modelDir,'solver_poly.prototxt') ;

opts.imageSize = [473, 473] ;
opts.numEpochs = 20 ;
opts.miniBatchSize = 1 ;
opts.valEpochs = 1 ; % val in every opts.valEpoch epoches
opts.useGpus = 0 ; % your GPU
opts = vl_argparse(opts, varargin);

% -------------------------------------------------------------------------
% Get data
% -------------------------------------------------------------------------
imdb = load(fullfile(opts.imdbDir, 'imdb.mat')) ;
stats = load(fullfile(opts.imdbDir, 'imdbStats.mat')) ;
train = find(imdb.images.set == 1 & imdb.images.segmentation) ;
val = find(imdb.images.set == 2 & imdb.images.segmentation) ;
opts.rgbMean_train = stats.rgbMean_train ;
opts.rgbMean_val = stats.rgbMean_val ;
opts.numClass = numel(imdb.objectClasses.id) ;

% -------------------------------------------------------------------------
% Set gpu
% -------------------------------------------------------------------------
if ~isempty(opts.useGpus)
  caffe.set_mode_gpu();
  caffe.set_device(opts.useGpus);
else
  caffe.set_mode_cpu();
end

% -------------------------------------------------------------------------
% Initialize model
% -------------------------------------------------------------------------
solver = caffe.Solver(model_solver);
solver.net.copy_from(model_weights);

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------
for epoch = 1 : opts.numEpochs
    
    % train
    if exist('train')
        numMinibatches = ceil(numel(train) / opts.miniBatchSize) ;
        inputsIndex = randperm(numel(train)) ;  % randomlize the training sequence

        
        pAccuracy = zeros(1,numel(train)) ;
        pCorrect = zeros(1,numel(train)) ;
        pLabeled = zeros(1,numel(train)) ;    
        areaIntersection = zeros(opts.numClass, numel(train)) ;
        areaUnion = zeros(opts.numClass, numel(train)) ; 
        bicls_pAccuracy = zeros(1,numel(train)) ;
        bicls_pCorrect = zeros(1,numel(train)) ;
        bicls_pLabeled = zeros(1,numel(train)) ;         
        
        for miniBatch = 1 : numMinibatches
            fprintf('Epoch%d: train %d/%d', epoch, miniBatch, numMinibatches) ;
            
            % generate minibatch training data
            if miniBatch == numMinibatches
                batchIdx = inputsIndex((miniBatch-1) * opts.miniBatchSize + 1 : end) ;
                numResIdx = opts.miniBatchSize - length(batchIdx) ;
                batchIdx = [batchIdx, inputsIndex(1 : numResIdx)] ;
            else
                batchIdx = inputsIndex((miniBatch-1) * opts.miniBatchSize + 1 : miniBatch * opts.miniBatchSize) ;
            end
            batch = train(batchIdx) ;           
            inputs_data = get_batch(imdb, batch, opts) ;

            % set inputs
            for n = 2:2:length(inputs_data)
                solver.net.blobs(solver.net.inputs{n/2}).set_data(inputs_data{n});
            end
              
            % one iter of SGD update
            time = tic ;
            solver.step(1);
            time = toc(time) ;
            
            % evaluate the accuracy
            prediction = solver.net.blobs('conv6_biclass_interp').get_data() ;
            prediction_pos = prediction(:, :, 2:2:end) ;
            prediction_neg = prediction(:, :, 1:2:end-1) ;
            bicls_pred = (prediction_pos - prediction_neg) > 0 ;
            bicls_pred = bicls_pred + 1 ;            
            bicls_anno = inputs_data{4} + 1 ;
            [bicls_pAccuracy(batchIdx), bicls_pCorrect(batchIdx), bicls_pLabeled(batchIdx)] = ...
                pixelAccuracy(bicls_pred, bicls_anno) ; 
            
            prediction = solver.net.blobs('conv6_interp').get_data() ;
            [~, prediction] = max(prediction,[],3 ) ;   
            prediction = uint8(prediction) ;
            anno = inputs_data{6} + 1 ;
            if strcmp(opts.dataset, 'pascal')
                anno(anno==255) = 0 ;
                anno(anno==256) = 0 ;
            end 
            prediction = imresize(prediction, size(anno), 'nearest') ;
            [pAccuracy(batchIdx), pCorrect(batchIdx), pLabeled(batchIdx)] = pixelAccuracy(prediction, anno) ;
            [areaIntersection(:,batchIdx), areaUnion(:,batchIdx)]=intersectionAndUnion(prediction, anno, opts.numClass);
            
            
            %print
            fprintf('||segAcc: %.3f', pAccuracy(batchIdx)) ;
            fprintf('||biclsAcc: %.3f', bicls_pAccuracy(batchIdx)) ; 
            speed = opts.miniBatchSize / time ;
            fprintf('||speed: %.1fHz\n', speed) ;
        end
        trainMeanPixelAccuracy = sum(pCorrect)/sum(pLabeled);
        trainMeanbiclsAccuracy = sum(bicls_pCorrect)/sum(bicls_pLabeled);
        IoU = sum(areaIntersection,2)./sum(eps+areaUnion,2);
        trainMeanIoU = mean(IoU);

        fprintf('for all images ||pixelAccuracy: %.3f', trainMeanPixelAccuracy) ;
        fprintf('  ||meanIoU: %.3f\n', trainMeanIoU) ;
    end
    
    % val
    if exist('val', 'var') && ...
       ((epoch == opts.numEpochs) || (mod(epoch, opts.valEpochs) == 0) )
   
        numMinibatches = ceil(numel(val) / opts.miniBatchSize) ; 

        pAccuracy = zeros(1,numel(val)) ;
        pCorrect = zeros(1,numel(val)) ;
        pLabeled = zeros(1,numel(val)) ;   
        areaIntersection = zeros(opts.numClass, numel(val)) ;
        areaUnion = zeros(opts.numClass, numel(val)) ;
        bicls_pAccuracy = zeros(1,numel(val)) ;
        bicls_pCorrect = zeros(1,numel(val)) ;
        bicls_pLabeled = zeros(1,numel(val)) ;        
        for miniBatch = 1 : numMinibatches
            fprintf('Epoch%d: val %d/%d', epoch, miniBatch, numMinibatches) ;
            
            if miniBatch == numMinibatches
                batchIdx = val((miniBatch-1) * opts.miniBatchSize + 1 : end) ;
                numResIdx = opts.miniBatchSize - length(batchIdx) ;
                batchIdx = [batchIdx, val(1 : numResIdx)] ;
            else
                batchIdx = val((miniBatch-1) * opts.miniBatchSize + 1 : miniBatch * opts.miniBatchSize) ;
            end
            batch = batchIdx ;
            
            inputs_data = get_batch(imdb, batch, opts) ;
            
            % set inputs
            inputs = {};
            for n = 2:2:length(inputs_data)
                inputs = cat(2,inputs,inputs_data{n});
            end
            
            % forward 
            time = tic ;
            tmp = solver.net.forward(inputs) ;
            time = toc(time) ;
            
            % evaluate the accuracy
            prediction = solver.net.blobs('conv6_biclass_interp').get_data() ;
            prediction_pos = prediction(:, :, 2:2:end) ;
            prediction_neg = prediction(:, :, 1:2:end-1) ;
            bicls_pred = (prediction_pos - prediction_neg) > 0 ;
            bicls_pred = bicls_pred + 1 ;            
            bicls_anno = inputs_data{4} + 1 ;
            [bicls_pAccuracy(batchIdx), bicls_pCorrect(batchIdx), bicls_pLabeled(batchIdx)] = ...
                pixelAccuracy(bicls_pred, bicls_anno) ;            
            
            prediction = solver.net.blobs('conv6_interp').get_data() ;
            [~, prediction] = max(prediction,[],3 ) ;  
            prediction = uint8(prediction) ;
            anno = inputs_data{6} + 1 ;
            if strcmp(opts.dataset, 'pascal')
                anno(anno==255) = 0 ;
                anno(anno==256) = 0 ;
            end             
            prediction = imresize(prediction, size(anno), 'nearest') ;
            [pAccuracy(batchIdx), pCorrect(batchIdx), pLabeled(batchIdx)] = pixelAccuracy(prediction, anno) ;
            [areaIntersection(:,batchIdx), areaUnion(:,batchIdx)]=intersectionAndUnion(prediction, anno, opts.numClass);  
            
            %print
            fprintf('||segAcc: %.3f', pAccuracy(batchIdx)) ;
            fprintf('||biclsAcc: %.3f', bicls_pAccuracy(batchIdx)) ;            
            speed = opts.miniBatchSize / time ;
            fprintf('||speed: %.1fHz\n', speed) ;
        end
        valMeanPixelAccuracy = sum(pCorrect)/sum(pLabeled);
        valMeanbiclsAccuracy = sum(bicls_pCorrect)/sum(bicls_pLabeled);        
        IoU = sum(areaIntersection,2)./sum(eps+areaUnion,2);
        valMeanIoU = mean(IoU);

        fprintf('for all images ||pixelAccuracy: %.3f', valMeanPixelAccuracy) ;
        fprintf('  ||meanIoU: %.3f\n', valMeanIoU) ;
    end
    
    % save net
    netPath = fullfile(opts.modelDir, ['net-epoch-',num2str(epoch),'.caffemodel']) ;
    solver.net.save(netPath) ;
    statePath = fullfile(opts.modelDir, ['state-epoch-',num2str(epoch),'.mat']) ;
    if exist('val', 'var') && ((epoch == opts.numEpochs) || (mod(epoch, opts.valEpochs) == 0) ) 
            save(statePath, 'trainMeanPixelAccuracy', 'trainMeanIoU', 'trainMeanbiclsAccuracy', ...
                            'valMeanPixelAccuracy', 'valMeanIoU', 'valMeanbiclsAccuracy') ;
    else
            save(statePath, 'trainMeanPixelAccuracy', 'trainMeanIoU', 'trainMeanbiclsAccuracy') ; 
    end
end

caffe.reset_all();



