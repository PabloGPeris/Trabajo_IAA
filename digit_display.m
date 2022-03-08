function digit_display(image, k)
digito = zeros(28,28);

for i=1:28
    for j=1:28
        digito(i,j)=image((i-1)*28+j,k);
    end
end
imshow(digito, 'DisplayRange', [0 255]);

end