%% Trabajo Inteligencia Artificial Aplicada

%% Bucle de Bayes para PCA
clear

addpath("..\")
load Trainnumbers.mat % para la clasificación básicamente
load("datos_PCA.mat", "data_pca"); % sin hacer PCA previa

%% Datos
% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

% k-means para el LDA (con valor a 0, se usa PCA; si no, se usa LDA)
k = 0;

% máx PCA
P = 50;

%% LDA en vez de PCA
if k
    load("datos_normalizacion.mat") %#ok<*UNRCH> 
    [new_data, new_label] = clustering_kmeans(data_n, Trainnumbers.label, k);
                
    sep_data = class_separation(new_data, new_label);
    
    [~, SW, SB, ~] = scatter_matrices(sep_data);
    
    % se usa la pseudoinversa para el cálculo de la matriz
    [coeff_lda,latent_lda] = eig(pinv(SW)*SB, 'vector'); % coeff_lda = W_lda
    [latent_lda, ind] = sort(latent_lda, 'descend');
    coeff_lda = real(coeff_lda(:, ind(1:10*k-1)));
    
    data_pca = (coeff_lda'*data_n)'; % truco -> renombra

    P = min(P, 10*k-1);
end

%% PCA
% nº datos
N = length(Trainnumbers.label); 

accuracy_PCA = zeros(P, 1);
accuracy = zeros(10, 1);
time_train_PCA = zeros(P, 1);
time_train = zeros(10, 1);
time_class_PCA = zeros(P, 1);
time_class = zeros(10, 1);

for j = 1:P
    for i = 1:10
        %% PCA previa (nº de dimensiones)
        % coge solo las dimensiones requeridas en la PCA
        data_r_pca = data_pca(:, 1:j)';
    
        %% Separar datos en train y test aleatoriamente
        % los datos se mezclan (permutan y se separan)
        ind_random = randperm(N);
        
        % train
        data_train = data_r_pca(:, ind_random(1:round(N*PD)));
        label_train = Trainnumbers.label(ind_random(1:round(N*PD)));
        
        % test
        data_test = data_r_pca(:, ind_random(round(N*PD)+1:end));
        label_test = Trainnumbers.label(ind_random(round(N*PD)+1:end));

        %% Clasificador bayesiano
        % train
        tic
        bayesModel = fitcnb(data_train', label_train', 'Prior', ones(1, 10*k));
        time_train(i) = toc;
        
        % test (classification)
        tic
        label_pred = predict(bayesModel, data_test')';
        time_class(i) = toc;

        accuracy(i) = sum(label_test == label_pred)/round(N*(1-PD));
    end
    accuracy_PCA(j) = mean(accuracy);
    time_train_PCA(j) = mean(time_train);
    time_class_PCA(j) = mean(time_class);

    disp("PCA" + num2str(j) + "/" + num2str(P) + " - Acc: " + num2str(accuracy_PCA(j)))
end

%% Figuras
figure(12);
plot(1:P, accuracy_PCA*100, 'LineWidth', 1.5);
xlabel('PCA')
ylabel('Accuracy (%)')
% legend('ccuracy')
grid on

figure(13);
plot(1:P, time_train_PCA*1000, 1:P, time_class_PCA*1000, 'LineWidth', 1.5);
xlabel('PCA')
ylabel('Time (ms)')
legend('Training', 'Classification')
grid on