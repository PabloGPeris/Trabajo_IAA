%% Trabajo Inteligencia Artificial Aplicada
%% Múltiples Bayes para hacer pruebas
clear

addpath("..\")
load Trainnumbers.mat % para la clasificación básicamente
load("datos_PCA.mat", "data_pca"); % sin hacer PCA previa

%% Datos
% dimensiones de la PCA
PCA = 20;

% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;

% número de iteraciones en el bucle
I = 10;

% k-means
k = 5;

%% PCA
data_r_pca = data_pca(:, 1:PCA)'; 
% otra forma -> data_r_pca = coeff_pca(:,1:D_pca)'*data_n ;

%% k - means
[new_data, new_label] = clustering_kmeans(data_r_pca, Trainnumbers.label, k);

% nº datos
N = length(Trainnumbers.label); 

accuracy = 0;
conf_mat = zeros(10, 10);

% %% PCA previa (nº de dimensiones)
% % coge solo las dimensiones requeridas en la PCA
% data_r_pca = data_pca(:, 1:PCA)';

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
            bayesModel = fitcnb(data_train', label_train', 'Prior', ones(1, 10*k), ...
                'Cost', Mcost);

%             bayesModel = fitcnb(data_train', label_train', 'Prior', ones(1, 10*k));
        catch ME
            errores = true;
        end
    end

    %     bayesModel = fitcnb(data_train', label_train', 'Prior', ones(1, 10), ...
    %         'OptimizeHyperparameters', 'auto');
    %     bayesModel = fitcnb(data_train', label_train', 'Prior', ones(1, 10), ...
    %         'DistributionNames', 'kernel', 'Kernel', 'triangle' );
    %     bayesModel = fitcnb(data_train', label_train', 'Prior', ones(1, 10), ...
    %         'DistributionNames', {'mvnm','mvmn'});

    % test (classification)
    label_pred = predict(bayesModel, data_test')';
    label_pred = mod(label_pred, 10);

    accuracy = accuracy + sum(label_test == label_pred)/round(N*(1-PD));
    conf_mat = conf_mat + confusionmat(label_test, label_pred);

    disp("iteration " + num2str(i) + "/" + num2str(I))
end

accuracy = accuracy / I

%% Figuras
figure(13);
confusionchart(conf_mat, 0:9, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');

figure(14);
confusionchart(conf_mat, 0:9, ...
    'ColumnSummary','absolute', ...
    'RowSummary','absolute');

%% Clasificación - Errores
% n = 8;
% d_pca = [15 16];
% 
% ind_TP_U_FP = label_pred == n; % predicted positive
% ind_TP_U_FN = label_test == n; % is positive
% ind_TP = ind_TP_U_FP & ind_TP_U_FN; % true positive
% ind_FP = ind_TP_U_FP & ~ind_TP; % false positive
% ind_FN = ind_TP_U_FN & ~ind_TP; % false negative
% 
% figure(13);
% plot(data_train(d_pca(1),ind_TP), data_train(d_pca(2),ind_TP), 'x', 'LineWidth', 1.5)
% hold on
% plot(data_train(d_pca(1),ind_FP), data_train(d_pca(2),ind_FP), 'x', 'LineWidth', 1.5)
% plot(data_train(d_pca(1),ind_FN), data_train(d_pca(2),ind_FN), 'x', 'LineWidth', 1.5)
% hold off
% xlabel('PCA 1')
% ylabel('PCA 2')
% legend('TP', 'FP', 'FN')