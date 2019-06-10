% copy from VOC toolkit


% VOCLABELCOLORMAP Creates a label color map such that adjacent indices have different
% colors.  Useful for reading and writing index images which contain large indices,
% by encoding them as RGB images.
%
% CMAP = VOCLABELCOLORMAP(N) creates a label color map with N entries.

%% config
N = 60 ;% the number of classes for the dataset, including background

%% generate
cmap = zeros(N,3);
for i=1:N
    id = i-1; r=0;g=0;b=0;
    for j=0:7
        r = bitor(r, bitshift(bitget(id,1),7 - j));
        g = bitor(g, bitshift(bitget(id,2),7 - j));
        b = bitor(b, bitshift(bitget(id,3),7 - j));
        id = bitshift(id,-3);
    end
    cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
% cmap = cmap / 255;

colors = uint8(cmap(2:end, :)) ; % exclude background

save(['color', num2str(N - 1), '.mat'], 'colors') ;

% save object color
w = 150 ;
h = 30 ;

color_dir = ['color', num2str(N - 1)] ;

imdb = load('imdb.mat') ;
object_name = imdb.objectClasses.name ; 


if ~exist(color_dir) 
    mkdir(color_dir) ;
end

for i=1:size(cmap,1)
    color_map = ones(h,w,3) ;
    color_map(:,:,1) = color_map(:,:,1) .* cmap(i, 1) ;
    color_map(:,:,2) = color_map(:,:,2) .* cmap(i, 2) ;
    color_map(:,:,3) = color_map(:,:,3) .* cmap(i, 3) ;
    
    imwrite(uint8(color_map), fullfile(color_dir, [object_name{i}, '.png'])) ; 
end
