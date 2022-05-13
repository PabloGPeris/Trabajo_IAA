clear all;
addpath("..\")
load Trainnumbers.mat

PD = 0.8;

% número de iteraciones en el bucle
I = 3; % antes estaba en 10, lo he bajao a 3 pa que tarde menos

%% PCA
% nº datos
N = length(Trainnumbers.label);
accuracy = 0;
conf_mat = zeros(10, 10);

% Creamos red neuronal
net = feedforwardnet([10 20 6], 'traingd'); % Creamos red neuronal con tres capas con 2 neuronas, 3 neuronas y 2

for i = 1:I

    % Separar datos en train y test aleatoriamente
    % los datos se mezclan (permutan y se separan)
    ind_random = randperm(N);
    
    % train
    data_train = Trainnumbers.image(:, ind_random(1:round(N*PD)));
    label_train = Trainnumbers.label(ind_random(1:round(N*PD)));
    
    matrix_label_train = digits2matrix(label_train);

    % test
    data_test = Trainnumbers.image(:, ind_random(round(N*PD)+1:end));
    label_test = Trainnumbers.label(ind_random(round(N*PD)+1:end));
    
    matrix_label_test = digits2matrix(label_test);

    %% Multilayer Perceptron
    
    net = train(net, data_train, matrix_label_train);
    
    label_pred = net(data_test);
    
    % test (classification)
%     perf1 = perform(net, data_test, matrix_label_test);

    accuracy = accuracy + sum(matrix_label_test == label_pred)/round(N*(1-PD));
%     conf_mat = conf_mat + confusionmat(label_test, label_pred);

    disp("iteration " + num2str(i) + "/" + num2str(I))
end


% disp(net.numInputs)
% disp(net.numLayers)
% disp(net.numOutputs)
% disp(net.numWeightElements)
% disp(net.biasConnect)
% disp(net.inputConnect)
% disp(net.layerConnect)
% disp(net.outputConnect)
% disp(net.layers)
% disp(net.biases)
% disp(net.outputs)
% x = net.layers{1};

% Dividir el accuracy acumulado entre el numero de iteraciones
disp(net.numInputs)
disp(net.numLayers)
disp(net.numOutputs)
disp(net.numWeightElements)
disp(net.biasConnect)
disp(net.inputConnect)
disp(net.layerConnect)
disp(net.outputConnect)
disp(net.layers)
disp(net.biases)
disp(net.outputs)

x = net.layers{1}

accuracy = accuracy / I
%% Figuras
% figure(11);