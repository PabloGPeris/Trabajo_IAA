%% Trabajo Inteligencia Artificial Aplicada
clear
close all

addpath("..\")
load Trainnumbers.mat

%% Datos
% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

% número de iteraciones en el bucle
I = 3;

% nº datos
N = length(Trainnumbers.label); 

accuracy = 0;
conf_mat = zeros(10, 10);

% Creamos red neuronal. Ej: [2 6 4] -> son 3 capas con 2, 6 y 4 neuronas
net = feedforwardnet([10 20 6], 'traingd');


%%
for i = 1:I
    
    % Separar datos en train y test aleatoriamente
    % los datos se mezclan (permutan y se separan)
    ind_random = randperm(N);
    
    % train data
    data_train = Trainnumbers.image(:, ind_random(1:round(N*PD)));
    label_train = Trainnumbers.label(ind_random(1:round(N*PD)));
    
    % pasamos las etiquetas a matriz para que haya 10 salidas
    matrix_label_train = digits2matrix(label_train);
    
    % test data
    data_test = Trainnumbers.image(:, ind_random(round(N*PD)+1:end));
    label_test = Trainnumbers.label(ind_random(round(N*PD)+1:end));
    
    % pasamos las etiquetas a matriz para que haya 10 salidas
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
%     conf_mat = conf_mat + confusionmat(label_test, label_pred);
    
    disp("iteration " + num2str(i) + "/" + num2str(I))
end

% Dividir el accuracy acumulado entre el numero de iteraciones
accuracy = accuracy / I;
disp ("accuracy: ")
disp(accuracy*100)