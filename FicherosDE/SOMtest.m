%% SOM

clearvars
clc
close all

addpath("..\")
load Trainnumbers.mat
load("datos_PCA.mat", "data_pca"); % sin hacer PCA previa
load("datos_PCA_test.mat", "data_pca_test");

etiquetas = Trainnumbers.label;
PCA = 25;

% Creamos mapa SOM bidimensional
anchura = 30;
altura = 36;
n_neuronas = anchura*altura;

% PCA previa solo nº dimensiones requeridas en la PCA
data_r_pca = data_pca(:, 1:PCA)';
data_r_pca_test = data_pca_test(:, 1:PCA)';

%% Cargar SOM previamente entrenado y validado con 80/20

load SOM_entrenado;

%% Asignar valor (una etiqueta) a cada neurona
p_sep = class_separation(data_r_pca, etiquetas); % función mía

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

evaluacion = net(data_r_pca_test);
neuronas_activadas_test = vec2ind(evaluacion);

% Mira a qué clase pertenece esa neurona, y lo asigna
clase_predicha = clase(neuronas_activadas_test) - 1;

%% Guardado

class = clase_predicha;
PCA = 25;
name = {'LuisBF', 'PabloGP', 'JavierDM'};

save('Group1_som.mat', "name", "PCA", "class")

%% evaluacion con labels man
load label_man.mat

conf_mat = confusion(label_man, clase_predicha(:,1:240));
figure(3)
conf_chart = confusionchart(label_man, clase_predicha(:,1:240));

% Calcular accuracy
accuracy = trace(conf_chart.NormalizedValues/240);
disp(accuracy*100)

