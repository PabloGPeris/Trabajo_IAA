%% Trabajo Inteligencia Artificial Aplicada
% Multilayer Perceptron by PABLO

clear

addpath("..\")
load Trainnumbers.mat
load("datos_PCA.mat", "data_pca"); % sin hacer PCA previa

%% Datos
% dimensiones de la PCA
PCA = 100;
% PCA = "LDA";
% k = 3; % k-means del LDA

% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

% GPU
GPU = 'yes';

% número de iteraciones en el bucle
I = 1;

% capas
netLayers = [30 30 30];
activationFunction = "tansig"; % help nntransfer
% activationFunction = "logsig"; 

%% Red
net = feedforwardnet(netLayers);
% net = feedforwardnet(netLayers, 'traingd');

for i = 1:length(netLayers)
    net.layers{i}.transferFcn = activationFunction;
end
net.layers{length(netLayers)+1}.transferFcn = "softmax";

net.performFcn = "mse";
% net.trainParam.epochs =50;

% view(net)

N = length(Trainnumbers.label); 

%% PCA o LDA
if isstring(PCA) && PCA == "LDA"
    load datos_normalizacion.mat
    
    % k - means
    [new_data, new_label] = clustering_kmeans(data_n, Trainnumbers.label, k);
    
    % LDA
    sep_data = class_separation(new_data, new_label);

    [~, SW, SB, ~] = scatter_matrices(sep_data);

    % se usa la pseudoinversa para el cálculo de la matriz
    [coeff_lda,latent_lda] = eig(pinv(SW)*SB, 'vector'); % coeff_lda = W_lda
    [latent_lda, ind] = sort(latent_lda, 'descend');
    coeff_lda = real(coeff_lda(:, ind(1:10*k-1)));

    data_r_pca = coeff_lda'*data_n;
else
    % PCA
    data_r_pca = data_pca(:, 1:PCA)';
end

%% Entrenar red
max_accuracy = 0;
accuracy = 0;

for i = 1:I
    
    % Separar datos en train y test aleatoriamente
    % los datos se mezclan (permutan y se separan)
    ind_random = randperm(N);
    
    % train data
    data_train = data_r_pca(:, ind_random(1:round(N*PD)));
    label_train = Trainnumbers.label(ind_random(1:round(N*PD)));
    
    % test data
    data_test = data_r_pca(:, ind_random(round(N*PD)+1:end));
    label_test = Trainnumbers.label(ind_random(round(N*PD)+1:end));
    
    output_train = full(ind2vec(label_train + 1, 10));
    output_test = full(ind2vec(label_test + 1, 10));
    
    % train
    trained_net = train(net, data_train, output_train, 'useGPU', GPU);
    
    % prediction
    output_pred = trained_net(data_test);
    label_pred = vec2ind(output_pred) - 1;
    
    % performance (MSE)
    perf = perform(trained_net, output_test, output_pred)
    
    % matriz de confusión
    figure(91);
    conf_mat = confusionchart(label_test, label_pred);
    
    accuracy_i = trace(conf_mat.NormalizedValues)/round(N*(1-PD))
    accuracy = accuracy + accuracy_i;
    max_accuracy = max([max_accuracy accuracy_i]);

    disp("iteration " + num2str(i) + "/" + num2str(I))
end

%%
% Dividir el accuracy acumulado entre el numero de iteraciones
accuracy = accuracy / I;
disp ("accuracy: " + num2str(accuracy*100) + " máx: " + num2str(max_accuracy*100))

%%
save LDA3 accuracy


