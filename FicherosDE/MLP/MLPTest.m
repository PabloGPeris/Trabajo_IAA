%% Trabajo Inteligencia Artificial Aplicada
%% MLP test
clear

load("..\Preprocesamiento\datos_normalizacion_test.mat")
load("..\Preprocesamiento\Test_numbers_HW1.mat")
load("MLPnet.mat")

%% Test
output_pred = trained_net(data_n_test);
class = vec2ind(output_pred) - 1;

%% Guardado
PCA = height(data_n_test);

name = {'LuisBF', 'PabloGP', 'JavierDM'};
save('Group1_mlp.mat', "name", "PCA", "class")