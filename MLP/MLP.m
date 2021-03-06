%% Trabajo Inteligencia Artificial Aplicada
% Multilayer Perceptron

clear
close all

addpath("..\")
load Trainnumbers.mat
load("datos_PCA.mat", "data_pca"); % sin hacer PCA previa

%% Datos
% dimensiones de la PCA
PCA = 30;

% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

% número de iteraciones en el bucle
I = 2;

% nº datos
N = length(Trainnumbers.label); 

accuracy = 0;
conf_mat = zeros(10, 10);

% Creamos red neuronal. Ej: [2 6 4] -> son 3 capas con 2, 6 y 4 neuronas
% la de luis: [10 20 6]
% la mejor de javi hasta el momento [4 4 4]
net = patternnet([10 8 6], 'traingdm');
net.trainParam.epochs = 7;
net.trainParam.min_grad = 1e-4;
% net.trainParam

%% PCA previa (nº de dimensiones)
% coge solo las dimensiones requeridas en la PCA
data_r_pca = data_pca(:, 1:PCA)';

%%
for i = 1:I
    
    % Separar datos en train y test aleatoriamente
    % los datos se mezclan (permutan y se separan)
    ind_random = randperm(N);
    
    % train data
    data_train = data_r_pca(:, ind_random(1:round(N*PD)));
    label_train = Trainnumbers.label(ind_random(1:round(N*PD)));
    
    % Pasamos las etiquetas a matriz para que haya 10 salidas
    matrix_label_train = digits2matrix(label_train);
    
    % test data
    data_test = data_r_pca(:, ind_random(round(N*PD)+1:end));
    label_test = Trainnumbers.label(ind_random(round(N*PD)+1:end));
    
    % Pasamos las etiquetas a matriz para que haya 10 salidas
    matrix_label_test = digits2matrix(label_test);
    
    % Entrenamos la red
    net = train(net, data_train, matrix_label_train);
    
    % Predicciones de la red
    label_pred = net(data_test);
    label_pred2 = vec2ind(label_pred);
    label_pred2 = label_pred2 - 1;
    
    % test (classification)
%     perf1 = perform(net, data_test, matrix_label_test);
    
    % Calcular el accuracy
    accuracy = accuracy + sum(label_test == label_pred2)/round(N*(1-PD));
     conf_mat = conf_mat + confusionmat(label_test, label_pred2);
    
    disp("iteration " + num2str(i) + "/" + num2str(I))
end

% Dividir el accuracy acumulado entre el numero de iteraciones
accuracy = accuracy / I;
disp ("accuracy: ")
disp(accuracy*100)

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