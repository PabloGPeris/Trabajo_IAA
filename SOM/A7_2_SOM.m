%% 7.2 - Self-Organizing Maps

clearvars
clc
close all

load data_D2_C2.mat
%% a) Entrenar SOM
n_neuronas = 10;

% crea un SOM
% net = selforgmap([5 5]);
% net = selforgmap([5 5], 100, 3, 'randtop', 'linkdist');
net = selforgmap([n_neuronas 1]);
net.trainParam.showWindow = 0; % 0 = Cierra ventana de visualizacion
net = train(net,p.valor);

% view(net)

figure(1)
plotsompos(net)
hold on
gscatter(p.valor(1,:), p.valor(2,:), p.clase, 'gc')
hold off

figure(2)
plotsomtop(net)
hold off

%% b) Asignar valor a cada neurona
p_sep = class_separation(p.valor, p.clase); % función mía

%%
n_apariciones = zeros(length(p_sep), n_neuronas); 

for i = 1:length(p_sep) % length(p_sep) = n_clases
    % mete los datos de cada clase y obtiene neuronas activadas
    neuronas_activadas = vec2ind(net(p_sep{i}));

    % mira cuántas veces se ha activado cada neurona
    n_apariciones(i,:) = sum(neuronas_activadas(:) == (1:n_neuronas));
end

% cada neurona será de la clase que más la haya activado
[~, clase] = max(n_apariciones)

%% c) Clasificar datos de test
% mete los datos de test y obtiene neuronas activadas
asdf = net(t.valor)
neuronas_activadas_test = vec2ind(net(t.valor));

% mira a qué clase pertenece esa neurona, y lo asigna
clase_predicha = clase(neuronas_activadas_test);

% matriz de confusión
figure(3)
conf_mat = confusionchart(t.clase, clase_predicha);

figure(4)
plotsompos(net)
hold on
gscatter(t.valor(1,:), t.valor(2,:), t.clase, 'gc')
hold off

%% b) Asignar valor a cada neurona con NN
% halla las neuronas activas (en unos y ceros) con los datos de preparación
neuronas_activadas_2 = net(p.valor);

% crea red cuya entrada es el SOM, y la salida son las clases
net2 = feedforwardnet([]);
net2.layers{1}.transferFcn = "logsig";
net2.trainParam.showWindow = 0;
net2 = train(net2, neuronas_activadas_2, full(ind2vec(p.clase)));

% view(net2)

% ¿qué asigna a cada clase
abduskan = net2(eye(n_neuronas))
[~, clase2] = max(abduskan)

%% c) Clasificar datos de test con NN
% mete los datos de test y obtiene neuronas activadas
[~, clase_predicha_2] = max(net2(net(t.valor))) 
% probablemente se puedan concatenar las NN de manera directa, pero no sé

% matriz de confusión
figure(5)
conf_mat_2 = confusionchart(t.clase, clase_predicha_2);
