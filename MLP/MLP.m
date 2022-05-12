%% Trabajo Inteligencia Artificial Aplicada
%% Múltiples Bayes para hacer pruebas
clear
close all

addpath("..\")
load Trainnumbers.mat % para la clasificación básicamente
load("datos_PCA.mat", "data_pca"); % sin hacer PCA previa

%% Datos
% dimensiones de la PCA
PCA = 20;

% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

% número de iteraciones en el bucle
I = 10;

%% PCA
% nº datos
N = length(Trainnumbers.label); 

accuracy = 0;
conf_mat = zeros(10, 10);

% Creamos red neuronal
net = feedforwardnet(4, 'traingd');
%view(net)
%net.layers{1}

%%

for i = 1:I
    %% PCA previa (nº de dimensiones)
    % coge solo las dimensiones requeridas en la PCA
    data_r_pca = data_pca(:, 1:PCA)';

    %% Separar datos en train y test aleatoriamente
    % los datos se mezclan (permutan y se separan)
    ind_random = randperm(N);
    
    % train
    data_train = data_r_pca(:, ind_random(1:round(N*PD)));
    label_train = Trainnumbers.label(ind_random(1:round(N*PD)));
    
    % test
    data_test = data_r_pca(:, ind_random(round(N*PD)+1:end));
    label_test = Trainnumbers.label(ind_random(round(N*PD)+1:end));

    %% Multilayer Perceptron

    net = train(net, data_train, label_train);
    
    label_pred = net(data_test);

    % test (classification)
    perf1 = perform(net, data_test, label_test);

    accuracy = accuracy + sum(label_test == label_pred)/round(N*(1-PD));
%     conf_mat = conf_mat + confusionmat(label_test, label_pred);

    disp("iteration " + num2str(i) + "/" + num2str(I))
end

view(net)
accuracy = accuracy / I

%% Figuras
% figure(11);
% confusionchart(conf_mat, 0:9, ...
%     'ColumnSummary','column-normalized', ...
%     'RowSummary','row-normalized');
% 
% figure(12);
% confusionchart(conf_mat, 0:9, ...
%     'ColumnSummary','absolute', ...
%     'RowSummary','absolute');

%% Clasificación - Errores
% n = 8;
% d_pca = [15 16];
% 
% ind_TP_U_FP = label_pred == n; % predicted positive
% ind_TP_U_FN = label_test == n; % is positive
% ind_TP = ind_TP_U_FP & ind_TP_U_FN; % true positive
% ind_FP = ind_TP_U_FP & ~ind_TP; % false positive
% ind_FN = ind_TP_U_FN & ~ind_TP; % false negative
% 
% figure(13);
% plot(data_train(d_pca(1),ind_TP), data_train(d_pca(2),ind_TP), 'x', 'LineWidth', 1.5)
% hold on
% plot(data_train(d_pca(1),ind_FP), data_train(d_pca(2),ind_FP), 'x', 'LineWidth', 1.5)
% plot(data_train(d_pca(1),ind_FN), data_train(d_pca(2),ind_FN), 'x', 'LineWidth', 1.5)
% hold off
% xlabel('PCA 1')
% ylabel('PCA 2')
% legend('TP', 'FP', 'FN')