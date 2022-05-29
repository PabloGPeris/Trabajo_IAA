function data_deepsaver2(image, x, N)


% Guardamos imagenes
for i = 1 : N

    data_test{i} = data_deepconverter(image, i, x);
    imwrite(data_test{i}, sprintf('../../../ImagenesTest/FIG%d.jpeg',i));

end

end