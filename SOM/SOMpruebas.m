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


% *** Parametros del selforgmap ***
% net = selforgmap (dimensions, coverSteps, initNeighbor, topologyFcn, distanceFcn);
% net = selforgmap([8 8], 100, 3, 'hextop', 'linkdist');

% *** topologyFcn ***
% randtop - una al azar
% hextop - hexagonal
% gridtop - cuadrada
% tritop - triangular

% *** distanceFcn ***
% linkdist - distancia de enlace (no se cual es)
% dist - distancia euclidea
% mandist - distancia Manhattan
% boxdist - distancia entre dos vectores posicion

anchura = 30;
altura = 36;
n_neuronas = anchura*altura;

% Crear SOM bidimensional

% Pruebas previas de estructura del mapa

% selforgmap([anchura altura]); 18x30 neuronas y 200 epoch -> accuracy = 0.864

% selforgmap([anchura altura]); 18x30 neuronas y 300 epoch -> accuracy = 0.855

% selforgmap([anchura altura], 100, 5, 'gridtop', 'mandist'); 18x30 neuronas
% y 200 epoch -> accuracy = 0.867

% selforgmap([anchura altura], 100, 5, 'gridtop', 'dist'); 18x30 neuronas 
% y 200 epoch -> accuracy = 0.857

% selforgmap([anchura altura], 100, 5, 'hextop', 'dist'); 18x30 neuronas y
% 200 epoch -> accuracy = 0.8745

% selforgmap([anchura altura], 100, 6, 'hextop', 'dist'); 18x30 neuronas y
% 200 epoch -> accuracy = 0.8705

% selforgmap([anchura altura], 100, 5, 'hextop', 'dist'); 36x30 neuronas y
% 200 epoch -> accuracy = 0.891

% selforgmap([anchura altura], 100, 5, 'hextop', 'dist'); 36x36 neuronas y
% 200 epoch -> accuracy = 0.8875


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

%% matriz de confusión

conf_mat = confusion(label_test, clase_predicha);
figure(3)
conf_chart = confusionchart(label_test, clase_predicha);

% Calcular accuracy
accuracy = trace(conf_chart.NormalizedValues)/(N*(1-PD));
disp(accuracy*100)

% Hacer el grafico accuracy frente a numero de epoch
% figure
% plot()

% conf_mat = confusionchart(neuronas_activadas, clase_predicha);

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
