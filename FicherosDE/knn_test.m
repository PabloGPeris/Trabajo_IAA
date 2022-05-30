%% KNN
addpath("..")
load Trainnumbers.mat
load("datos_PCA.mat", "data_pca");
load Preprocesamiento\datos_PCA_test

%% Parámetros
% dimensiones de la PCA
PCA = 45;

% Vecinos más cercanos
K = 3;

%% Train 

data_r_pca = data_pca(:, 1:PCA)';
data_train = data_r_pca;

label_train = Trainnumbers.label;

knnModel = fitcknn(data_train', label_train', 'Prior', ones(1, 10), 'NumNeighbors', K);


%% Test

data_r_pca_test = data_pca_test(:, 1:PCA)';  %Datos de test con PCA
data_test = data_r_pca_test;

class = predict(knnModel, data_test')';  % Etiquetas predichas

%% Guardado
<<<<<<< HEAD
name = {'LuisBF', 'PabloGM', 'JaviDM'};
PCA = 0;
=======
name = {'LuisBF', 'PabloGP', 'JavierDM'};
>>>>>>> dafc6f45f8f6a8bf41de64aed1e4052d48bd8288
save('Group1_knn2.mat', "name", "PCA", "class")



