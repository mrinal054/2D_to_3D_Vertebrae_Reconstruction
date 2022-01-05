% Create X-ray images for different projection angles
clc; clear all; close all; 

%% Data-1
matName = 'vertebra20';             % Name of the mat file
load(['resizedCT\', matName])
destPath1 = 'xray';
destPath2 = '1-deg-diff-projections-all-vertebrae-y-axis';

name = double(scaled);
name = rescale(name, 0, 255);       % Rescale color intensity in [0,255]
out_folder = matName;

% ----> Uncomment the following part if you want to resize data3d_2
% img_size = 128;
% data3d_1_resized = imresize(data3d, [img_size, img_size]);
% val = 1:3:224;
% data3d_1_compressed = data3d_1_resized(:,:,val);

rot = 0:1:360;            %intervals in rotation angles
axis = 'y_axis';         %axis along which you are rotating for naming
axis_rot = [0 1 0];       %define rotation axis

for i = 1:length(rot)
%   Choose data3d_2 or data3d_2_compressed
    B = imrotate3(name, rot(i), axis_rot,'cubic','crop','FillValues',0);
%---> Uncomment if you want avg x-ray
%     xRay = fnXrayU(B,0.15,5.5); 
    xRay = fnXray(B);
    xRay = uint8(xRay); 
%     figure; imshow(xRay, [])

%---> Uncomment if you want max x-ray    
%     xRay = fnXrayMax(B);
    
%     xRay = uint8(xRay);
%     xRay = uint8(rescale(xRay, 0, 255));    
%     xRay = (rescale(xRay, 0, 255));
%     xRay = uint8(rescale(xRay, 0, 255));
    
% ------> Uncomment if you want to resize X-ray data
%     xRayData1 = imresize(xRay, [img_size, img_size]);

        destPath = fullfile(destPath1, destPath2, out_folder);
        if ~exist(destPath, 'dir')
            mkdir(destPath);
        end
        
        %***** Uncomment to save in .mat format
%         if rot(i)<10            
%             save(fullfile(destPath, ...
%             [axis, '_deg_', '00', num2str(rot(i)),'.mat']), 'xRay');
%         elseif (rot(i)>=10 && rot(i)<100)
%             save(fullfile(destPath, ...
%             [axis, '_deg_', '0', num2str(rot(i)),'.mat']), 'xRay');
%         elseif rot(i)>=100
%             save(fullfile(destPath, ...
%             [axis, '_deg_', num2str(rot(i)),'.mat']), 'xRay');
%         end

        %***** Uncomment to save in .png         
        if rot(i)<10
            imwrite(xRay, fullfile(destPath, [axis, '_deg_', '00', num2str(rot(i)),'.png']));
        elseif (rot(i)>=10 && rot(i)<100)
            imwrite(xRay, fullfile(destPath, [axis, '_deg_', '0', num2str(rot(i)),'.png']));
        elseif rot(i)>=100
            imwrite(xRay, fullfile(destPath, [axis, '_deg_', num2str(rot(i)),'.png']));
        end
    

end


