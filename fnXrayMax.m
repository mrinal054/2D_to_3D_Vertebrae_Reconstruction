function out = fnXrayMax(img)
%{
input: CT image
output: 2D X-ray image
%}
    max_img = zeros(size(img,1), size(img,2));
    for i = 1:size(img,1)
        for j = 1:size(img,2)
            max_img(i,j) = max(img(i,j,:));
        end
    end
    out = max_img;
end

