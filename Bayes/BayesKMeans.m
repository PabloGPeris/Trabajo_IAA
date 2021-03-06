%% Trabajo Inteligencia Artificial Aplicada
%% Bucle de Bayes para k-means
clear

addpath("..\")
load Trainnumbers.mat % para la clasificación básicamente
load("datos_PCA.mat", "data_pca"); % sin hacer PCA previa

%% Datos
% dimensiones de la PCA
PCA = 20;
% PCA = "LDA"; % poner esto para que haga LDA en vez de PCA - resultados
% malos si se hace LDA y k-means con diferentes valores (en verdad, las
% divisiones del Batesiano debe ser MAYOR o igual que el LDA)

% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

% número de iteraciones en el bucle
I = 30;

% k-means
K = 3;

%% PCA
if isstring(PCA) && PCA == "LDA"
    load("datos_LDA.mat")
    load ("datos_normalizacion.mat")
    data_r_pca = coeff_lda'*data_n ; % en realidad
else
    data_r_pca = data_pca(:, 1:PCA)'; 
end

%% k-means
N = length(Trainnumbers.label); 

accuracy_KM = zeros(K, 1);
accuracy_KM_min = zeros(K, 1);
accuracy_KM_max = zeros(K, 1);
accuracy = zeros(I, 1);
time_train_KM = zeros(K, 1);
time_train = zeros(I, 1);
time_class_KM = zeros(K, 1);
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
            [new_data, new_label] = clustering_kmeans(data_r_pca, Trainnumbers.label, k);
            
            % Separar datos en train y test aleatoriamente
            % los datos se mezclan (permutan y se separan)
            ind_random = randperm(N);
            
            % train
            data_train = new_data(:, ind_random(1:round(N*PD)));
            label_train = new_label(ind_random(1:round(N*PD)));
            
            % test
            data_test = new_data(:, ind_random(round(N*PD)+1:end));
            label_test = new_label(ind_random(round(N*PD)+1:end));
            label_test = mod(label_test, 10); % se reducen cosas
    
            try
                %% Clasificador bayesiano
                % train
                tic
                bayesModel = fitcnb(data_train', label_train', 'Prior', ones(1, 10*k), ...
                    'Cost', Mcost);
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
    accuracy_KM(k) = mean(accuracy);
    accuracy_KM_min(k) = min(accuracy);
    accuracy_KM_max(k) = max(accuracy);
    time_train_KM(k) = mean(time_train);
    time_class_KM(k) = mean(time_class);

    disp("k-means k = " + num2str(k) + "/" +num2str(K) + " - Acc: " + ...
        num2str(accuracy_KM(k)) + " max(" + num2str(accuracy_KM_max(k))+ ")" )
end

save BayesKmeans.mat time_class_KM time_train_KM accuracy_KM accuracy_KM_min accuracy_KM_max

%% Figuras
figure(15);
plot(1:K, accuracy_KM*100, 'LineWidth', 1.5);
hold on
plot(1:K, accuracy_KM_min*100, '--', 'LineWidth', 1.5);
plot(1:K, accuracy_KM_max*100, '--', 'LineWidth', 1.5);
hold off
xlabel('k')
ylabel('Accuracy (%)')
legend('mean', 'min', 'max')
grid on

figure(16);
plot(1:K, time_train_KM*1000, 1:K, time_class_KM*1000, 'LineWidth', 1.5);
xlabel('k')
ylabel('Time (ms)')
legend('Training', 'Classification')
grid on



