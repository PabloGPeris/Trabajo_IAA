%% SOM

clearvars
clc
close all

addpath("..\")
load Trainnumbers.mat
load("datos_PCA.mat", "data_pca"); % sin hacer PCA previa

etiquetas = Trainnumbers.label;
PCA = 20;
PD = 0.8; % porcentaje (tanto por uno) de datos de training (80/20 tipico)
 
%%Crear SOM bidimensional
n_neuronas_mitad = 5;
n_neuronas = n_neuronas_mitad*2;

% net = selforgmap([5 5], 100, 3, 'randtop', 'linkdist');
net = selforgmap([n_neuronas_mitad 2]);
net.trainParam.epochs = 30;

% PCA previa (nº de dimensiones)
% coge solo las dimensiones requeridas en la PCA
data_r_pca = data_pca(:, 1:PCA)';

%% Separar datos en train y test
% nº datos
N = length(Trainnumbers.label);

% los datos se mezclan (permutan y se separan)
ind_random = randperm(N);

indices_separacion = ind_random(1:round(N*PD));

% train
data_train = data_r_pca(:, ind_random(1:round(N*PD)));
label_train = etiquetas(ind_random(1:round(N*PD)));

% test
data_test = data_r_pca(:, ind_random(round(N*PD)+1:end));
label_test = etiquetas(ind_random(round(N*PD)+1:end));


%% Entrenar SOM
net.trainParam.showWindow = 1; % 0 = Cierra ventana de visualizacion
net = train(net,data_train');

% view(net)

% figure(1)
% plotsompos(net)
% hold on
% gscatter(p.valor(1,:), p.valor(2,:), p.clase, 'gc')
% hold off

% figure(2)
% plotsomtop(net)
% hold off

%% Asignar valor a cada neurona
p_sep = class_separation(data_r_pca, etiquetas); % función mía
% p_sep = class_separation(data_train, label_train); % función mía
% p_sep2 = class_separation(data_r_pca, etiquetas);

%%
p_sep_junto = [];
%juntar los cells en una matriz
for i = 1:10
    p_sep_junto = [p_sep_junto p_sep{i}];
end

%%
p_sep_train = [];

% coger de p_sep, los datos de las posiciones random que hemos usado para
% separar en 80/20 los datos, y que asi salgan uniformes
for i = 1:10
    p_sep_junto_train = p_sep_junto(:, indices_separacion);
end

p_sep_train = class_separation(p_sep_junto_train, label_train)

%%

n_clases = length(p_sep_train);
n_apariciones = zeros(n_clases, n_neuronas);

for i = 1:n_clases
    % mete los datos de cada clase y obtiene neuronas activadas
    neuronas_activadas = vec2ind(net(p_sep_train{i}));
    
    % obtiene cuántas veces se ha activado cada neurona
    n_apariciones(i,:) = sum(neuronas_activadas(:) == (1:n_neuronas));
end

%% cada neurona será de la clase que más la haya activado
[~, clase] = max(n_apariciones)

%% c) Clasificar datos de test
% mete los datos de test y obtiene neuronas activadas
neuronas_activadas_test = vec2ind(net(t.valor));
% 
% % mira a qué clase pertenece esa neurona, y lo asigna
% clase_predicha = clase(neuronas_activadas_test);
% 
% % matriz de confusión
% figure(3)
% conf_mat = confusionchart(t.clase, clase_predicha);
% 
% figure(4)
% plotsompos(net)
% hold on
% gscatter(t.valor(1,:), t.valor(2,:), t.clase, 'gc')
% hold off
% 
% %% b) Asignar valor a cada neurona con NN
% % halla las neuronas activas (en unos y ceros) con los datos de preparación
% neuronas_activadas_2 = net(p.valor);
% 
% % crea red cuya entrada es el SOM, y la salida son las clases
% net2 = feedforwardnet([]);
% net2.layers{1}.transferFcn = "logsig";
% net2.trainParam.showWindow = 0;
% net2 = train(net2, neuronas_activadas_2, full(ind2vec(p.clase)));
% 
% % view(net2)
% 
% % ¿qué asigna a cada clase
% abduskan = net2(eye(n_neuronas))
% [~, clase2] = max(abduskan)
% 
% %% c) Clasificar datos de test con NN
% % mete los datos de test y obtiene neuronas activadas
% [~, clase_predicha_2] = max(net2(net(t.valor))) 
% % probablemente se puedan concatenar las NN de manera directa, pero no sé
% 
% % matriz de confusión
% figure(5)
% conf_mat_2 = confusionchart(t.clase, clase_predicha_2);
