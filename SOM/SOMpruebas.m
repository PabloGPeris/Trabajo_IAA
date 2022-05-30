%% SOM

clearvars
clc
close all

addpath("..\")
load Trainnumbers.mat
load("datos_PCA.mat", "data_pca"); % sin hacer PCA previa

etiquetas = Trainnumbers.label;
PCA = 25;
PD = 0.8; % porcentaje (tanto por uno) de datos de training (80/20 tipico)

% Crear un mapa SOM bidimensional
anchura = 30;
altura = 36;
n_neuronas = anchura*altura;

net = selforgmap([anchura altura], 100, 5, 'hextop', 'dist');
net.trainParam.epochs = 200;
net.trainParam.showWindow = 1; % 0 = Cierra ventana de visualizacion

% PCA previa solo nº dimensiones requeridas en la PCA
data_r_pca = data_pca(:, 1:PCA)';

%% Separacion 80/20 del data_r_pca

N = length(Trainnumbers.label); % nº datos
ind_random = randperm(N); % los datos se mezclan (permutan y se separan)

% datos de entrenamiento
data_train = data_r_pca(:, ind_random(1:round(N*PD)));
label_train = etiquetas(ind_random(1:round(N*PD)));

% datos de test
data_test = data_r_pca(:, ind_random(round(N*PD)+1:end));
label_test = etiquetas(ind_random(round(N*PD)+1:end));


%% Entrenar SOM
net = train(net,data_train);

%% Asignar valor (una etiqueta) a cada neurona
p_sep = class_separation(data_train, label_train); % función mía

n_clases = length(p_sep);
n_apariciones = zeros(n_clases, n_neuronas);

for i = 1:n_clases
    % mete los datos de cada clase y obtiene neuronas activadas
    neuronas_activadas = vec2ind(net(p_sep{i}));
    
    % obtiene cuántas veces se ha activado cada neurona
    n_apariciones(i,:) = sum(neuronas_activadas(:) == (1:n_neuronas));
end

% cada neurona será de la clase que más la haya activado
[~, clase] = max(n_apariciones)

%% Evaluacion de la red

evaluacion = net(data_test);
neuronas_activadas_test = vec2ind(evaluacion);

% Mira a qué clase pertenece esa neurona, y lo asigna
clase_predicha = clase(neuronas_activadas_test) - 1;

%% Matriz de confusión

conf_mat = confusion(label_test, clase_predicha);
figure(3)
conf_chart = confusionchart(label_test, clase_predicha);

% Calcular accuracy
accuracy = trace(conf_chart.NormalizedValues)/(N*(1-PD));
disp(accuracy*100)

%% Guardamos la red entrenada para probarla con los datos de test en el futuro
save SOM_entrenado net;
