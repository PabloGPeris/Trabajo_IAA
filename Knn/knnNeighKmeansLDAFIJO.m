

%% Cargamos los datos 
clear all;
addpath("..\")
load Trainnumbers.mat % para la clasificación básicamente
load("datos_normalizacion.mat")




%% Datos
% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

% número de iteraciones en el bucle
I = 10;

% k-means
K = 2;

% K-neigh
Kn = 3;

N = length(Trainnumbers.label); 

%% k - means
[new_data, new_label] = clustering_kmeans(data_n, Trainnumbers.label, K);

%% LDA
sep_data = class_separation(new_data, new_label);

[~, SW, SB, ~] = scatter_matrices(sep_data);

% se usa la pseudoinversa para el cálculo de la matriz
[coeff_lda,latent_lda] = eig(pinv(SW)*SB, 'vector'); % coeff_lda = W_lda
[latent_lda, ind] = sort(latent_lda, 'descend');
coeff_lda = real(coeff_lda(:, ind(1:10*K-1)));

new_data_lda = coeff_lda'*new_data;



%% Entrenar
% Separar datos en train y test aleatoriamente
% los datos se mezclan (permutan y se separan)
ind_random = randperm(N);

% train
data_train = new_data_lda(:, ind_random(1:round(N*PD)));
label_train = new_label(ind_random(1:round(N*PD)));

% test
data_test = new_data_lda(:, ind_random(round(N*PD)+1:end));
label_test = new_label(ind_random(round(N*PD)+1:end));

%%
tic
knnModel = fitcknn(data_train', label_train','NumNeighbors',Kn);
time_train = toc;


%%
tic
label_pred = predict(knnModel, data_test')';
time_class = toc;


accuracy = sum(label_test == label_pred)/round(N*(1-PD))







