function digit_display(image, k)
%DIGIT_DISPLAY(image,k)
%
% Dibuja el dígito k. Trainnumbers.image o similar es el que se debe usar.
digito = zeros(28,28);

for i=1:28
    for j=1:28
        digito(i,j)=image((i-1)*28+j,k);
    end
end
imshow(digito, 'DisplayRange', [0 255]);

end