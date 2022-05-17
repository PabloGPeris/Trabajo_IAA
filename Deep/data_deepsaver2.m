function data_deepsaver2(image, label, x, N, PD)

% A diferencia de la anterior esta guarda los dtos de train en diferentes
% carpetas dependiendo de la clase

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

% Guardamos imagenes train
for i = 1 : 10000*PD
     
    switch label_train(i)
        case 0
            imwrite(data_train{i}, sprintf('../../ImagenesDeep/Train/0/FIG%d.jpeg',i));
        case 1
            imwrite(data_train{i}, sprintf('../../ImagenesDeep/Train/1/FIG%d.jpeg',i));
        case 2
            imwrite(data_train{i}, sprintf('../../ImagenesDeep/Train/2/FIG%d.jpeg',i));
        case 3
            imwrite(data_train{i}, sprintf('../../ImagenesDeep/Train/3/FIG%d.jpeg',i));
        case 4
            imwrite(data_train{i}, sprintf('../../ImagenesDeep/Train/4/FIG%d.jpeg',i));
        case 5
            imwrite(data_train{i}, sprintf('../../ImagenesDeep/Train/5/FIG%d.jpeg',i));
        case 6
            imwrite(data_train{i}, sprintf('../../ImagenesDeep/Train/6/FIG%d.jpeg',i));
        case 7
            imwrite(data_train{i}, sprintf('../../ImagenesDeep/Train/7/FIG%d.jpeg',i));
        case 8
            imwrite(data_train{i}, sprintf('../../ImagenesDeep/Train/8/FIG%d.jpeg',i));
        case 9
            imwrite(data_train{i}, sprintf('../../ImagenesDeep/Train/9/FIG%d.jpeg',i));
        otherwise
            disp('Fallo data_saver2')
    end
    
end

% Guardamos las imagenes test
for i = 1 : round(10000*(-PD+1))

    imwrite(data_test{i}, sprintf('../../ImagenesDeep/Test/FIG%d.jpeg',i));
    
end

save('../../ImagenesDeep/label_train.mat',"label_train")

save('../../ImagenesDeep/data_test.mat',"data_test"); %Inecesario

save('../../ImagenesDeep/label_test.mat',"label_test")

end