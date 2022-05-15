%% Trabajo Inteligencia Artificial Aplicada
%% Bucle de Bayes para LDA
clear

addpath("..\")
load Trainnumbers.mat % para la clasificación básicamente
load("datos_normalizacion.mat")
% load("datos_PCA.mat", "data_pca"); % sin hacer PCA previa

%% Datos
% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

% número de iteraciones en el bucle
I = 20;

% k-means
K = 3;

%% k-means
N = length(Trainnumbers.label); 

accuracy_LDA = zeros(K, 1);
accuracy_LDA_min = zeros(K, 1);
accuracy_LDA_max = zeros(K, 1);
accuracy = zeros(I, 1);
time_train_LDA = zeros(K, 1);
time_train = zeros(I, 1);
time_class_LDA = zeros(K, 1);
time_class = zeros(I, 1);

for k = 1:K
    % matriz de costes
    Mcost1 = ones(10, 10) - eye(10, 10);
    Mcost = repmat(Mcost1, k, k);
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
                %% Clasificador bayesiano
                % train
                tic
                bayesModel = fitcnb(data_train', label_train', 'Prior', ones(1, 10*k), ...
                    'Cost', Mcost);
%                 bayesModel = fitcnb(data_train', label_train', 'Prior', ones(1, 10));
                time_train(i) = toc;
            catch ME
                errores = true;
            end
        end
        
        % test (classification)
        tic
        label_pred = predict(bayesModel, data_test')';
        label_pred = mod(label_pred, 10);
        time_class(i) = toc;
        label_pred = mod(label_pred, 10);

        accuracy(i) = sum(label_test == label_pred)/round(N*(1-PD));
        disp("Iteration " + num2str(i) + "/" + num2str(I));
    end
    accuracy_LDA(k) = mean(accuracy);
    accuracy_LDA_min(k) = min(accuracy);
    accuracy_LDA_max(k) = max(accuracy);
    time_train_LDA(k) = mean(time_train);
    time_class_LDA(k) = mean(time_class);

    disp("LDA with k-means k = " + num2str(k) + "/" +num2str(K) + " - Acc: " + ...
        num2str(accuracy_LDA(k)) + " max(" + num2str(accuracy_LDA_max(k))+ ")" )
end

save BayesLDA.mat time_class_LDA time_train_LDA accuracy_LDA accuracy_LDA_min accuracy_LDA_max

%% Figuras
figure(17);
plot(1:K, accuracy_LDA*100, 'LineWidth', 1.5);
hold on
plot(1:K, accuracy_LDA_min*100, '--', 'LineWidth', 1.5);
plot(1:K, accuracy_LDA_max*100, '--', 'LineWidth', 1.5);
hold off
xlabel('k')
ylabel('Accuracy (%)')
legend('mean', 'min', 'max')
grid on

figure(18);
plot(1:K, time_train_LDA*1000, 1:K, time_class_LDA*1000, 'LineWidth', 1.5);
xlabel('k')
ylabel('Time (ms)')
legend('Training', 'Classification')
grid on



