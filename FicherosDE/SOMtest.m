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
% PD = 0.8; % porcentaje (tanto por uno) de datos de training (80/20 tipico)


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

% CON PCA=25 ;selforgmap([anchura altura], 100, 5, 'hextop', 'dist'); 36x30 neuronas y
% 200 epoch -> accuracy = 0.90 


net = selforgmap([anchura altura], 100, 5, 'hextop', 'dist');
net.trainParam.epochs = 200;
net.trainParam.showWindow = 1; % 0 = Cierra ventana de visualizacion

% PCA previa solo nº dimensiones requeridas en la PCA
data_r_pca = data_pca(:, 1:PCA)';
data_r_pca_test = data_pca_test(:, 1:PCA)';

%% Separacion 80/20 del data_r_pca
% 
% N = length(Trainnumbers.label); % nº datos
% ind_random = randperm(N); % los datos se mezclan (permutan y se separan)
% 
% % datos de entrenamiento
% data_train = data_r_pca(:, ind_random(1:round(N*PD)));
% label_train = etiquetas(ind_random(1:round(N*PD)));
% 
% % datos de test
% data_test = data_r_pca(:, ind_random(round(N*PD)+1:end));
% label_test = etiquetas(ind_random(round(N*PD)+1:end));


%% Entrenar SOM
net = train(net,data_r_pca);


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

%% matriz de confusión

load label_man.mat

conf_mat = confusion(label_man, clase_predicha(:,1:240));
figure(3)
conf_chart = confusionchart(label_man, clase_predicha(:,1:240));

% Calcular accuracy
accuracy = trace(conf_chart.NormalizedValues/240);
disp(accuracy*100)


% conf_mat = confusionchart(neuronas_activadas, clase_predicha(:,1:110));

% figure(4)
% plotsompos(net)


%%

% matriznueva = []
% 
% sz = [anchura altura];
% [row,col] = ind2sub (sz, 1:1080);
% 
% %%
% for i = 1:n_neuronas
%     matriznueva(row(1,i),col(1,i)) = clase(1,i) - 1;
% end

%% Guardado

class = clase_predicha;
name = {'LuisBade', 'PabloPer', 'JaviDiaz'};
% PCA = 25;
save('Group1_som.mat', "name", "PCA", "class")


%%
% indices_digitos = [22,2,9,6,31,21,13,4,11,15]
% 
% figure
% hold on;
% for i = 1:n_neuronas
%     disp("iteracion: ")
%     disp(i)
%     % subploteamos la figura en 30x36 subplots
% %     fila = row(1,i);
% %     colu = col(1,i);
% 
%     subplot(anchura, altura, i);
% 
%     % este es el digito de la neurona i
%     digito_a_pintar = clase(:,i) - 1;
% 
%     digit_display(Trainnumbers.image, indices_digitos(digito_a_pintar+1))
% 
% end
% hold off;