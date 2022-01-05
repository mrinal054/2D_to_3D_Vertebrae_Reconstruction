function out = fnXray(img)
%{
input: .nii image
output: 2D X-ray image
%}
    sum_img = zeros(size(img,1), size(img,2));
    for i = 1:size(img,3)
        sum_img = sum_img + img(:,:,i); %sum-up accross all slices
    end        
    avg_img = sum_img./size(img,3); %take average
    out = avg_img;
end
    