function digito_convertido = data_deepconverter(image, k, x)
%
%   Dibuja el dígito k en dimensiones x,x. Trainnumbers.image o similar es el que se debe
%   usar.
digito = zeros(28,28);

for i=1:28
    for j=1:28
        digito(i,j)=image((i-1)*28+j,k);
    end
end
digito_convertido = imresize(digito,[x,x]);

grayImage = 255 * uint8(digito_convertido);
digito_convertido = cat(3, grayImage, grayImage, grayImage);

%imshow(digito_convertido, 'DisplayRange', [0 255]);

end