%% Trabajo Inteligencia Artificial Aplicada
%% Bayes - con LDA en test
clear

load("..\Preprocesamiento\datos_normalizacion_test.mat")
load("..\Preprocesamiento\Test_numbers_HW1.mat")
load("bayesModel.mat")

% load("..\..\datos_normalizacion.mat")
% load("..\..\Trainnumbers.mat")

%% Test
data_lda = coeff_lda'*data_n_test;

class = predict(bayesModel, data_lda')';  % Etiquetas predichas
class = mod(class, 10); % las etiquetas son 0, 1, ..., 9, 10 = 0, 11 = 1, ...

%% Guardado
PCA = 19;

name = {'LuisBF', 'PabloGP', 'JavierDM'};
save('Group1_bay.mat', "name", "PCA", "class")