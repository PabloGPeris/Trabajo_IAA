%% Trabajo Inteligencia Artificial Aplicada
%% First Bayes
% Aquí se entrena el clasificador bayesiano - esto de llamarle a todo
% FirstAlgo quizás debería parar - llamarle trainBayes o algo así
clear

addpath("..\")
load Trainnumbers.mat % para la clasificación básicamente
load("datos_PCA.mat", "data_pca", "latent"); % sin hacer PCA previa
% load("datos_PCA.mat", "data_pca_r"); % data_r_pca es lo interesante

%% Datos
% dimensiones de la PCA
PCA = 20;

% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

%% PCA previa (nº de dimensiones)
% coge solo las dimensiones requeridas en la PCA
data_r_pca = data_pca(:, 1:PCA)';

% MSE esperado
MSE_esperado = (sum(latent) - cumsum(latent))/sum(latent);
MSE = MSE_esperado(PCA)

%% Separar datos en train y test
% nº datos
N = length(Trainnumbers.label); 

% los datos se mezclan (permutan y se separan)
ind_random = randperm(N);

% train
data_train = data_r_pca(:, ind_random(1:round(N*PD)));
label_train = Trainnumbers.label(ind_random(1:round(N*PD)));

% test
data_test = data_r_pca(:, ind_random(round(N*PD)+1:end));
label_test = Trainnumbers.label(ind_random(round(N*PD)+1:end));

%% Clasificador bayesiano
% train
bayesModel = fitcnb(data_train', label_train', 'Prior', ones(1, 10));

% test
label_pred = predict(bayesModel, data_test')';

% matriz de confusión
figure(11);
conf_mat = confusionchart(label_test, label_pred);

accuracy = trace(conf_mat.NormalizedValues)/round(N*(1-PD))

save Bayes.mat bayesModel PCA
