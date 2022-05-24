%% Cargamos los datos 
clear all;
addpath("..\")
load Trainnumbers.mat % para la clasificación básicamente
load("datos_normalizacion.mat")


%% Realizamos el kmeans al LDA
% k = 5;
% 
% [data_lda, D_lda_max] = KmeansLDAfunction(k);


%% Datos
% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

% número de iteraciones en el bucle
I = 10;

% k-means
K = 5;

% K-neigh
Kn = 5;

%% k-means
N = length(Trainnumbers.label); 

accuracy_LDA = zeros(K, Kn);
accuracy_LDA_min = zeros(K, Kn);
accuracy_LDA_max = zeros(K, Kn);
accuracy = zeros(I,Kn);
time_train_LDA = zeros(K, Kn);
time_train = zeros(I, Kn);
time_class_LDA = zeros(K, Kn);
time_class = zeros(I, Kn);

for k = 1:K
    for jKn = 1:Kn
        for i = 1:I
            errores = true;
            while errores
                errores = false;

                %% k - means
                [new_data, new_label] = clustering_kmeans(data_n, Trainnumbers.label, k);

                %% LDA
                sep_data = class_separation(new_data, new_label);

                [~, SW, SB, ~] = scatter_matrices(sep_data);

                % se usa la pseudoinversa para el cálculo de la matriz
                [coeff_lda,latent_lda] = eig(pinv(SW)*SB, 'vector'); % coeff_lda = W_lda
                [latent_lda, ind] = sort(latent_lda, 'descend');
                coeff_lda = real(coeff_lda(:, ind(1:10*k-1)));

                new_data_lda = coeff_lda'*new_data;

                %             data_lda = coeff_lda'*data_n;

                %% Entrenar
                % Separar datos en train y test aleatoriamente
                % los datos se mezclan (permutan y se separan)
                ind_random = randperm(N);

                % train
                data_train = new_data_lda(:, ind_random(1:round(N*PD)));
                label_train = new_label(ind_random(1:round(N*PD)));

                % test
                data_test = new_data_lda(:, ind_random(round(N*PD)+1:end));
                label_test = new_label(ind_random(round(N*PD)+1:end));
                label_test = mod(label_test, 10); % se reducen cosas

                %             % train
                %             data_train = data_lda(:, ind_random(1:round(N*PD)));
                %             label_train = Trainnumbers.label(ind_random(1:round(N*PD)));
                %
                %             % test
                %             data_test = data_lda(:, ind_random(round(N*PD)+1:end));
                %             label_test = Trainnumbers.label(ind_random(round(N*PD)+1:end));


                try
                    %% Clasificador knn
                    % train
                    tic
                    knnModel = fitcknn(data_train', label_train', 'Prior', ones(1, 10*k),'NumNeighbors',K);
                    time_train(i,jKn) = toc;
                catch ME
                    errores = true;
                end
            end

            % test (classification)
            tic
            label_pred = predict(knnModel, data_test')';
            label_pred = mod(label_pred, 10);
            time_class(i,jKn) = toc;
            label_pred = mod(label_pred, 10);

            accuracy(i,jKn) = sum(label_test == label_pred)/round(N*(1-PD));
            disp("Iteration " + num2str(i) + "/" + num2str(I));
        end
        accuracy_LDA(k,jKn) = mean(accuracy(:,jKn));
        accuracy_LDA_min(k,jKn) = min(accuracy(:,jKn));
        accuracy_LDA_max(k,jKn) = max(accuracy(:,jKn));
        time_train_LDA(k,jKn) = mean(time_train(:,jKn));
        time_class_LDA(k,jKn) = mean(time_class(:,jKn));

        disp("LDA with k-means k = " + num2str(k) + "/" +num2str(K) + "k kneigh "+ num2str(jKn) + "/" +num2str(Kn)+" - Acc: " + ...
            num2str(accuracy_LDA(k)) + " max(" + num2str(accuracy_LDA_max(k))+ ")" )
    end
end

%% Figuras
figure(17);

% plot3(1:K, 1:Kn, accuracy_LDA*100, 'LineWidth', 1.5);

plot(1:Kn, accuracy_LDA(1,:)*100, 'LineWidth', 1.5);
hold on
plot(1:Kn, accuracy_LDA(2,:)*100, 'LineWidth', 1.5);
hold on
plot(1:Kn, accuracy_LDA(3,:)*100, 'LineWidth', 1.5);
hold on
plot(1:Kn, accuracy_LDA(4,:)*100, 'LineWidth', 1.5);
hold on
plot(1:Kn, accuracy_LDA(5,:)*100, 'LineWidth', 1.5);
hold on
xlabel('k-neigh')
ylabel('Accuracy (%)')
legend('kmeans1', 'kmeans2', 'kmeans3','kmeans4','kmeans5')
grid on
%%
% figure(18);
% plot(1:K, time_train_LDA*1000, 1:K, time_class_LDA*1000, 'LineWidth', 1.5);
% xlabel('k')
% ylabel('Time (ms)')
% legend('Training', 'Classification')
% grid on
