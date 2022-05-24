%% Cargamos los datos de la LDA
clear all;                                                              %Conclusión: A mayor numero de dimens mejor, o sorpresa

addpath("..\")
load Trainnumbers.mat % para la clasificación básicamente
load("datos_LDA.mat", "data_lda"); 

%% Datos
% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

%% PCA
% nº datos
N = length(Trainnumbers.label); 

accuracy_LDA = zeros(9, 1);
accuracy = zeros(10, 1);
time_train_LDA = zeros(9, 1);
time_train = zeros(10, 1);
time_class_LDA = zeros(9, 1);
time_class = zeros(10, 1);

for j = 1:9
    for i = 1:10
        %% LDA previa (nº de dimensiones)
        % coge solo las dimensiones requeridas en la PCA
        data_r_lda = data_lda(:, 1:j)';
    
        %% Separar datos en train y test aleatoriamente
        % los datos se mezclan (permutan y se separan)
        ind_random = randperm(N);
        
        % train
        data_train = data_r_lda(:, ind_random(1:round(N*PD)));
        label_train = Trainnumbers.label(ind_random(1:round(N*PD)));
        
        % test
        data_test = data_r_lda(:, ind_random(round(N*PD)+1:end));
        label_test = Trainnumbers.label(ind_random(round(N*PD)+1:end));

        %% Clasificador knn
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
    accuracy_LDA(j) = mean(accuracy);
    time_train_LDA(j) = mean(time_train);
    time_class_LDA(j) = mean(time_class);

    disp("LDA" + num2str(j) + "/100 - Acc: " + num2str(accuracy_LDA(j)))
end

% Para ver la dimension con mejores resultados y la acc de esta
[max_accu, position] = max(accuracy_LDA)

%% Figuras
figure(12);
plot(1:9, accuracy_LDA*100, 'LineWidth', 1.5);
xlabel('LDA')
ylabel('Accuracy (%)')
% legend('Accuracy')
grid on

%%
figure(13);
plot(1:9, time_train_LDA*1000, 1:9, time_class_LDA*1000, 'LineWidth', 1.5);
xlabel('LDA')
ylabel('Time (ms)')
legend('Training', 'Classification')
grid on