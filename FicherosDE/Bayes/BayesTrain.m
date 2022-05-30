%% Trabajo Inteligencia Artificial Aplicada
%% Bayes - con LDA
clear

addpath("..\..\")

load("Trainnumbers.mat") % para la clasificación básicamente
load("datos_normalizacion.mat")
load("datos_PCA.mat", "data_pca"); % sin hacer PCA previa

%% Datos
% dimensiones de la PCA
% PCA = 20;
PCA = "LDA";

% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

% k-means previo
k = 2;

if isstring(PCA) && PCA == "LDA"
    [new_data, new_label] = clustering_kmeans_v2(data_n, Trainnumbers.label, k);
    
    %% LDA
    sep_data = class_separation(new_data, new_label);
    [~, SW, SB, ~] = scatter_matrices(sep_data);
    % se usa la pseudoinversa para el cálculo de la matriz
    [coeff_lda,latent_lda] = eig(pinv(SW)*SB, 'vector'); % coeff_lda = W_lda
    [latent_lda, ind] = sort(latent_lda, 'descend');
    coeff_lda = real(coeff_lda(:, ind(1:10*k-1)));

    new_data = coeff_lda'*new_data;
else
    %% PCA
    data_r_pca = data_pca(:, 1:PCA)'; 

    [new_data, new_label] = clustering_kmeans_v2(data_n, Trainnumbers.label, k);
end

%% k - means
% nº datos
N = length(Trainnumbers.label); 

% matriz de costes
Mcost1 = ones(10, 10) - eye(10, 10);
Mcost = repmat(Mcost1, k, k);


%% k - means    
% Separar datos en train y test aleatoriamente
% los datos se mezclan (permutan y se separan)
ind_random = randperm(N);

% train
data_train = new_data(:, ind_random(1:round(N*PD)));
label_train = new_label(ind_random(1:round(N*PD)));

% test
data_test = new_data(:, ind_random(round(N*PD)+1:end));
label_test = new_label(ind_random(round(N*PD)+1:end));
label_test = mod(label_test, 10); % se reducen cosas

try
    %% Clasificador bayesiano
    % train
    bayesModel = fitcnb(data_train', label_train', 'Prior', ones(1, 10*k), ...
        'Cost', Mcost);

catch ME
    disp(ME)
    BayesTrain
    return
end

% test (classification)
label_pred = predict(bayesModel, data_test')';
label_pred = mod(label_pred, 10);

accuracy = sum(label_test == label_pred)/round(N*(1-PD))
conf_mat = confusionmat(label_test, label_pred);

%% Figuras
figure(13);
confusionchart(conf_mat, 0:9, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');

figure(14);
confusionchart(conf_mat, 0:9, ...
    'ColumnSummary','absolute', ...
    'RowSummary','absolute');

%% Guardado
% save bayesModel.mat bayesModel coeff_lda

