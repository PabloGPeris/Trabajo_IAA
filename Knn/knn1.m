clear
%% Comprobamos los resultados con un PCA de PCA dimensiones                         Objetivo
addpath("..\")
load Trainnumbers.mat % para la clasificación básicamente
load("datos_PCA.mat", "data_pca", "latent"); % sin hacer PCA previa
% load("datos_PCA.mat", "data_pca_r"); % data_r_pca es lo interesante

%% Datos
% dimensiones de la PCA
PCA = 46;                                           %El que mejores resultados da según knnPCA

% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

% número de iteraciones en el bucle
I = 10;
%% PCA previa (nº de dimensiones)
% coge solo las dimensiones requeridas en la PCA
data_r_pca = data_pca(:, 1:PCA)';

% MSE esperado
MSE_esperado = (sum(latent) - cumsum(latent))/sum(latent);
MSE = MSE_esperado(PCA)

%% PCA
% nº datos
N = length(Trainnumbers.label); 

accuracy = 0;
conf_mat = zeros(10, 10);

%% PCA previa (nº de dimensiones)
% coge solo las dimensiones requeridas en la PCA
data_r_pca = data_pca(:, 1:PCA)';

for i = 1:I
    

    %% Separar datos en train y test aleatoriamente
    % los datos se mezclan (permutan y se separan)
    ind_random = randperm(N);
    
    % train
    data_train = data_r_pca(:, ind_random(1:round(N*PD)));
    label_train = Trainnumbers.label(ind_random(1:round(N*PD)));
    
    % test
    data_test = data_r_pca(:, ind_random(round(N*PD)+1:end));
    label_test = Trainnumbers.label(ind_random(round(N*PD)+1:end));

    %% Clasificador knn
    % train
    knnModel = fitcknn(data_train', label_train', 'Prior', ones(1, 10));
    % test (classification)
    label_pred = predict(knnModel, data_test')';

    accuracy = accuracy + sum(label_test == label_pred)/round(N*(1-PD));
    conf_mat = conf_mat + confusionmat(label_test, label_pred);

    disp("iteration " + num2str(i) + "/" + num2str(I))
end

accuracy = accuracy / I

%% Figuras
figure(11);
confusionchart(conf_mat, 0:9, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');

figure(12);
confusionchart(conf_mat, 0:9, ...
    'ColumnSummary','absolute', ...
    'RowSummary','absolute');

