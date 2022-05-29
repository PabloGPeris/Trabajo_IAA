%% Esto se va a quedar haciendo toda la noche

addpath("..\")
load Trainnumbers.mat
load("datos_PCA.mat", "data_pca"); % sin hacer PCA previa


results = {};
PD = 0.8;


for c1 = 20:10:40 
for c2 = 10:10:30
for c3 = 10:10:30
for PCA = 20:10:40
for act_function = ["logsig", "tansig"]
    % Parámetros
    netLayers = [c1 c2 c3];
    activationFunction = act_function;

    % PCA
    data_r_pca = data_pca(:, 1:PCA)';

    % Red
    net = feedforwardnet(netLayers, 'trainscg');
    
    for i = 1:length(netLayers)
        net.layers{i}.transferFcn = activationFunction;
    end
    net.layers{length(netLayers)+1}.transferFcn = "softmax";
    
    net.performFcn = "mse";

    N = length(Trainnumbers.label); 

%     net.trainParam.epochs = 120;
    net.trainParam.showWindow = false;

    % Preparar datos
    % Separar datos en train y test aleatoriamente
    % los datos se mezclan (permutan y se separan)
    ind_random = randperm(N);
    
    % train data
    data_train = data_r_pca(:, ind_random(1:round(N*PD)));
    label_train = Trainnumbers.label(ind_random(1:round(N*PD)));
    
    % test data
    data_test = data_r_pca(:, ind_random(round(N*PD)+1:end));
    label_test = Trainnumbers.label(ind_random(round(N*PD)+1:end));
    
    output_train = full(ind2vec(label_train + 1, 10));
    output_test = full(ind2vec(label_test + 1, 10));
    
    % train
    trained_net = train(net, data_train, output_train, 'useGPU','yes');
    
    % prediction
    output_pred = trained_net(data_test);
    label_pred = vec2ind(output_pred) - 1;

    % performance (MSE)
    perf = perform(net, output_test, output_pred);
    accuracy = sum(label_test == label_pred)/round(N*(1-PD));

    % Resultados
    results = [results; 
        {c1, c2, c3, PCA, act_function, perf, accuracy}];

    disp(num2str(c1) + ", " + num2str(c2) + ", " + num2str(c3) + ...
        ", PCA:" + num2str(PCA) + ", " + act_function + "; MSE:" + ...
        num2str(perf) + ", acc:" + num2str(accuracy*100));
end 
end
end
end
end

%%
res_T = cell2table(results, 'VariableNames', ...
    {'c1', 'c2', 'c3', 'PCA', 'fun', 'MSE', 'acc'});





