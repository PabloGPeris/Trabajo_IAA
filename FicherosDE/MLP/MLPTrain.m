%% Trabajo Inteligencia Artificial Aplicada
%% MLP
clear

addpath("..\..\")
load("Trainnumbers.mat") % para la clasificación básicamente
load("datos_normalizacion.mat")

%% Datos
% dimensiones de la PCA - no hay PCA
PCA = length(ind_validos); % 673 datos no nulos;

% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

% GPU
GPU = 'yes';

% capas
netLayers = [200 150 50];
activationFunction = "tansig"; % help nntransfer 
%% Red
net = feedforwardnet(netLayers);
for i = 1:length(netLayers)
    net.layers{i}.transferFcn = activationFunction;
end
net.layers{length(netLayers)+1}.transferFcn = "softmax";

net.performFcn = "mse";
net.input.processFcns = {'mapminmax'}; % si no, da error
net.output.processFcns = {'mapminmax'};

N = length(Trainnumbers.label); 

%% Entrenar red
% Separar datos en train y test aleatoriamente
% los datos se mezclan (permutan y se separan)
ind_random = randperm(N);

% train data
data_train = data_n(:, ind_random(1:round(N*PD)));
label_train = Trainnumbers.label(ind_random(1:round(N*PD)));

% test data
data_test = data_n(:, ind_random(round(N*PD)+1:end));
label_test = Trainnumbers.label(ind_random(round(N*PD)+1:end));

output_train = full(ind2vec(label_train + 1, 10));
output_test = full(ind2vec(label_test + 1, 10));

% train
trained_net = train(net, data_train, output_train, 'useGPU', GPU);

%% Validación
% prediction
output_pred = trained_net(data_test);
label_pred = vec2ind(output_pred) - 1;

% performance (MSE)
perf = perform(trained_net, output_test, output_pred)

% matriz de confusión
conf_mat = confusionmat(label_test, label_pred);

figure(91);
confusionchart(conf_mat, ...
'ColumnSummary','column-normalized', ...
'RowSummary','row-normalized');

accuracy = trace(conf_mat)/round(N*(1-PD))

%% Guardado
save MLPnet trained_net

