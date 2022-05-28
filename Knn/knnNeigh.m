%% Vemos con que dimensiones de la PCA se obtienen mejores valores                              Objetivo
%% Parece que cada vez un k de mejores resultados     0.25 cambio entre las accu                Conclusion
clear all;

addpath("..\")
load Trainnumbers.mat % para la clasificación básicamente
load("datos_PCA.mat", "data_pca"); % sin hacer PCA previa

%% Datos
% dimensiones de la PCA
PCA = 45;                                           %El que mejores resultados da según knnPCA

% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

% número de iteraciones en el bucle
I = 10;

%número de k´s vecinos más cercanos queremos probar 
K = 10;

%% PCA previa (nº de dimensiones)
% coge solo las dimensiones requeridas en la PCA
data_r_pca = data_pca(:, 1:PCA)';

% MSE esperado
%MSE_esperado = (sum(latent) - cumsum(latent))/sum(latent);
%MSE = MSE_esperado(PCA)

%% K's con PCA
N = length(Trainnumbers.label);

accuracy_Neigh = zeros(K, 1);
accuracy = zeros(10, 1);
time_train_Neigh = zeros(K, 1);
time_train = zeros(10, 1);
time_class_Neigh = zeros(K, 1);
time_class = zeros(10, 1);

for j = 1:K
    for i = 1:I

        %% Separar datos en train y test aleatoriamente
        % los datos se mezclan (permutan y se separan)
        ind_random = randperm(N);

        % train
        data_train = data_r_pca(:, ind_random(1:round(N*PD)));
        label_train = Trainnumbers.label(ind_random(1:round(N*PD)));

        % test
        data_test = data_r_pca(:, ind_random(round(N*PD)+1:end));
        label_test = Trainnumbers.label(ind_random(round(N*PD)+1:end));

        %% Clasificador knn
        % train
        tic
        knnModel = fitcknn(data_train', label_train', 'Prior', ones(1, 10),'NumNeighbors',j);
        time_train(i) = toc;

        % test (classification)
        tic
        label_pred = predict(knnModel, data_test')';
        time_class(i) = toc;

        accuracy(i) = sum(label_test == label_pred)/round(N*(1-PD));
    end
    accuracy_Neigh(j) = mean(accuracy);
    time_train_Neigh(j) = mean(time_train);
    time_class_Neigh(j) = mean(time_class);

    disp("K" + num2str(j) + "/10 - Acc: " + num2str(accuracy_Neigh(j)))
end
% Para ver la dimension con mejores resultados y la acc de esta
[max_accu, position] = max(accuracy_Neigh)

%% Figuras
figure(12);
plot(1:K, accuracy_Neigh*100, 'LineWidth', 1.5);
xlabel('Nº de vecinos (K)')
ylabel('Accuracy (%)')
% legend('ccuracy')
grid on

figure(13);
plot(1:K, time_train_Neigh*1000, 1:K, time_class_Neigh*1000, 'LineWidth', 1.5);
xlabel('Nº de vecinos (K)')
ylabel('Time (ms)')
legend('Training', 'Classification')
grid on