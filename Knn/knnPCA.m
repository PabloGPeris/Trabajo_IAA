

%% Vemos con que dimensiones de la PCA se obtienen mejores valores                              Objetivo
clear

addpath("..\")
load Trainnumbers.mat % para la clasificación básicamente
load("datos_PCA.mat", "data_pca"); % sin hacer PCA previa

%% Datos
% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

%% PCA
% nº datos
N = length(Trainnumbers.label); 

accuracy_PCA = zeros(100, 1);
accuracy = zeros(10, 1);
time_train_PCA = zeros(100, 1);
time_train = zeros(10, 1);
time_class_PCA = zeros(100, 1);
time_class = zeros(10, 1);

for j = 1:100
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
        knnModel = fitcknn(data_train', label_train', 'Prior', ones(1, 10));
        time_train(i) = toc;
        
        % test (classification)
        tic
        label_pred = predict(knnModel, data_test')';
        time_class(i) = toc;

        accuracy(i) = sum(label_test == label_pred)/round(N*(1-PD));
    end
    accuracy_PCA(j) = mean(accuracy);
    time_train_PCA(j) = mean(time_train);
    time_class_PCA(j) = mean(time_class);

    disp("PCA" + num2str(j) + "/100 - Acc: " + num2str(accuracy_PCA(j)))
end

% Para ver la dimension con mejores resultados y la acc de esta
[max_accu, position] = max(accuracy_PCA)

%% Figuras
figure(12);
plot(1:100, accuracy_PCA*100, 'LineWidth', 1.5);
xlabel('PCA')
ylabel('Accuracy (%)')
% legend('ccuracy')
grid on

figure(13);
plot(1:100, time_train_PCA*1000, 1:100, time_class_PCA*1000, 'LineWidth', 1.5);
xlabel('PCA')
ylabel('Time (ms)')
legend('Training', 'Classification')
grid on