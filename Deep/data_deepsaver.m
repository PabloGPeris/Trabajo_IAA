function data_deepsaver(image, label, x, N, PD)


%image son las imagenes que le pasamos
%x son las dimensiones que queremos para las imagenes

% Se guarda en formato png


% Separar datos en train y test aleatoriamente
% los datos se mezclan (permutan y se separan)
ind_random = randperm(N);

for i = 1 : 10000

    imagen_convertida{i} = data_deepconverter(image, i, x);

end

% train
data_train = imagen_convertida(:, ind_random(1:round(N*PD)));
label_train = label(ind_random(1:round(N*PD)));

% test
data_test = imagen_convertida(:, ind_random(round(N*PD)+1:end));
label_test = label(ind_random(round(N*PD)+1:end));


for i = 1 : 10000*PD

    % Guardamos las imagenes train
    imwrite(data_train{i}, sprintf('../../ImagenesDeep/Train/FIG%d.png',i));
    
end

for i = 1 : 10000*(-PD+1)

    % Guardamos las imagenes test
    imwrite(data_test{i}, sprintf('../../ImagenesDeep/Test/FIG%d.png',i));
    
end



save('../../ImagenesDeep/label_train.mat',"label_train")

save('../../ImagenesDeep/label_test.mat',"label_train")

end



%prueba = Imagenes_convertidas{1}  %Lo devuelve de la manera buena



