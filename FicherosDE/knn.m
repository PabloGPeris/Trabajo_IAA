%% Cargamos datos

clear all;

addpath("..\")
load Trainnumbers.mat % para la clasificación básicamente
load("datos_PCA.mat", "data_pca");

% Cargor los datos de test con la PCA aplicada

%% Parámetros

% dimensiones de la PCA
PCA = 45;

% Vecinos más cercanos
K = 3;


%% 

%Train
data_r_pca = data_pca(:, 1:PCA)';
data_train = data_r_pca;
label_train = Trainnumbers.label;

knnModel = fitcknn(data_train', label_train', 'Prior', ones(1, 10),'NumNeighbors',K);


%Test
data_r_pca_test = data_pca_test(:, 1:PCA)';  %Datos de test con PCA
data_test = data_r_pca_test;



label_pred = predict(knnModel, data_test')';





