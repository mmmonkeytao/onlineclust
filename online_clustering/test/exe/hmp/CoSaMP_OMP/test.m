clear all;


im = imread('cell_phone_1_1_6_crop.png');

% count  = 0;
% for i = 1 : size(im,1)
%       for j = 1 : size(im,2)
%            for k = 1:size(im,3)
%               if im(i,j,k) ~= 0
%                  count = count + 1;
%               end
%            end
%       end
% end
% count

patchsz_width = 5;
patchsz_height = 5;
stepsz_width = 5;
stepsz_height = 5;

hf_w = floor(patchsz_width / 2);
hf_h = floor(patchsz_height / 2);

im_szx = size(im,1);
im_szy = size(im,2);

patchMat(75, :) = 0;

patchth = 1;

    i = 1;
    while i <= im_szx
        
        j = 1;
        while j <= im_szy
            
            px = i -  hf_w;
            py = j - hf_h;
            
            for l = 0:patchsz_height-1
                for m = 0:patchsz_width-1
                    
                    cpx = px + l;
                    cpy = py + m;
                    
                    if cpx >0 && cpx < im_szx && cpy > 0 && cpy < im_szy        
                        patchMat( m+1 + l *patchsz_height, patchth ) = im(cpx ,cpy ,1);
                        patchMat( m+1 + l *patchsz_height + patchsz_width*patchsz_height, patchth ) = im(cpx,cpy ,2);
                        patchMat( m+1 + l *patchsz_height + 2*patchsz_width*patchsz_height, patchth) = im( cpx ,cpy ,3);
                    else
                        patchMat( m+1 + l *patchsz_height , patchth) = 0;
                        patchMat( m+1 + l *patchsz_height + patchsz_width*patchsz_height , patchth) = 0;
                        patchMat( m+1 + l *patchsz_height + 2*patchsz_width*patchsz_height, patchth) = 0;
                    end
                    
                end
            end
            
            patchth = patchth +1;
            j = j + stepsz_width;    
            
        end
        
        i = i + stepsz_height;
        
    end

% count  = 0;
% for i = 1 : size(patchMat,1)
%       for j = 1:size(patchMat,2)
%           if patchMat(i,j) ~= 0
%               count = count + 1;
%           end
%       end
% end
% count

patchMat = patchMat/255;

size(patchMat)
%D = load('dic_1st_layer_5x5_rgbcrop.dat');
D = rand(75,20000);
X = [];
for i = 1: size(patchMat, 2)
    [x,r] = OMP( D, patchMat(:,i), 15);
    X = [X x];
end
    
norm(patchMat - D*X)
    
    
    
    
    
    