%%% Trabajo Inteligencia Artificial Aplicada
%% Múltiples Bayes para hacer pruebas
clear

addpath("..\")
load Trainnumbers.mat % para la clasificación básicamente
load("datos_normalizacion.mat")
load("datos_PCA.mat", "data_pca"); % sin hacer PCA previa
load("datos_LDA.mat")

%% Datos
% dimensiones del autoencoder
hiddenSize = 25;

% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

% número de iteraciones en el bucle
I = 20;

% % k-means
% k = 1;

%% Autoencoder
% autoenc = trainAutoencoder(data_n, hiddenSize);

% MLP for compression (output=input)
% autoenc = newff(minmax(data_n), [floor((height(data_n) + hiddenSize)/2), ...
%     hiddenSize, floor((height(data_n) + hiddenSize)/2), height(data_n)], ...
%     {'tansig' 'purelin' 'tansig' 'purelin'}, 'trainlm');
% [autoenc,tr]=train(autoenc,data_n,data_n, 'useGPU', 'yes');
% 

load autoencP

% The first half of the NN for data compression
netcompr = newff(minmax(data_n),[floor((height(data_n) + hiddenSize)/2), ...
hiddenSize], {'tansig' 'purelin'},'trainlm');
netcompr.IW{1}=autoenc.IW{1}; 
netcompr.LW{2,1}=autoenc.LW{2,1};
netcompr.b{1}=autoenc.b{1}; 
netcompr.b{2}=autoenc.b{2};

% % The seconf half of the NN for data de-compression
% netdescompr=newff(minmax(p_compr),[floor((height(data_n) + hiddenSize)/2) ...
%       ,(height(data_n)],{'t
% ansig' 'purelin'}, 'trainlm');
% netdescompr.IW{1}=autoenc.LW{3,2}; netdescompr.LW{2,1}=net.LW{


%%

%% k - means
% nº datos
N = length(Trainnumbers.label); 

accuracy = 0;
conf_mat = zeros(10, 10);

% %% PCA previa (nº de dimensiones)
% % coge solo las dimensiones requeridas en la PCA
% data_r_pca = data_pca(:, 1:PCA)';
data_r_auto = netcompr(data_n);

for i = 1:I
    errores = true;
    while errores 
        errores = false;
        % Separar datos en train y test aleatoriamente
        % los datos se mezclan (permutan y se separan)
        ind_random = randperm(N);
        
        % train
        data_train = data_r_auto(:, ind_random(1:round(N*PD)));
        label_train = Trainnumbers.label(ind_random(1:round(N*PD)));
        
        % test
        data_test = data_r_auto(:, ind_random(round(N*PD)+1:end));
        label_test = Trainnumbers.label(ind_random(round(N*PD)+1:end));
        
        try
            %% Clasificador bayesiano
            % train
            bayesModel = fitcnb(data_train', label_train', 'Prior', ones(1, 10));

%             bayesModel = fitcnb(data_train', label_train', 'Prior', ones(1, 10*k));
        catch ME
            errores = true;
        end
    end

    % test (classification)
    label_pred = predict(bayesModel, data_test')';

    accuracy = accuracy + sum(label_test == label_pred)/round(N*(1-PD));
    conf_mat = conf_mat + confusionmat(label_test, label_pred);

    disp("iteration " + num2str(i) + "/" + num2str(I))
end

accuracy = accuracy / I

%% Figuras
figure(111);
confusionchart(conf_mat, 0:9, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');

figure(112);
confusionchart(conf_mat, 0:9, ...
    'ColumnSummary','absolute', ...
    'RowSummary','absolute');