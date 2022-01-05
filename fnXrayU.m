function out = fnXrayU(img,uw,d)
    
    sumRay = zeros(size(img,1), size(img,2));
%     I0 = img(:,:,1);
    I0 = 1;
%     eMat = zeros(size(img,1), size(img,2));
%     d = 0.1;
    
    for x = 1:size(img,3)
        HUx = img(:,:,x);
        ux = ((HUx/1000)+1)*uw;
        sumRay = sumRay + exp(ux*d);
    end
    out = I0*(sumRay);
end

   

