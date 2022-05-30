

load Trainnumbers.mat
load("datos_PCA.mat", "data_pca");
load datos_PCA_test

%% Parámetros
% dimensiones de la PCA
PCA = 45;

% Vecinos más cercanos
K = 3;

%% Train 

data_r_pca = data_pca(:, 1:PCA)';
data_train = data_r_pca;

label_train = Trainnumbers.label;

knnModel = fitcknn(data_train', label_train', 'Prior', ones(1, 10),'NumNeighbors',K);


%% Test

data_r_pca_test = data_pca_test(:, 1:PCA)';  %Datos de test con PCA
data_test = data_r_pca_test;

class = predict(knnModel, data_test')';  % Etiquetas predichas

%% Guardado
name = {'LuisBF', 'PabloGM', 'JaviDM'};
PCA = 0;
save('Group1_knn2.mat', "name", "PCA", "class")



