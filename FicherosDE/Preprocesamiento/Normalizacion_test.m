%% Trabajo Inteligencia Artificial Aplicada
%% Normalization Test
% Normalización test

clear
load("Test_numbers_HW1.mat")
load("..\..\datos_normalizacion.mat", "sigma_validos", "media_validos", "ind_validos");

%% Normalización
% se eliminan aquellos datos con sigma 0
data_test = Test_numbers.image(ind_validos,:);

% datos normalizados
data_n_test = ((data_test - media_validos)./sigma_validos);

% guardado de datos
save datos_normalizacion_test data_n_test