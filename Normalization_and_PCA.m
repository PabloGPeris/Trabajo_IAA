%% Trabajo Inteligencia Artificial Aplicada
clear
load Trainnumbers.mat

%% Variables
porcentaje =0.95;


%% Normalizar
sigma = std(Trainnumbers.image, 0, 2);
media = mean(Trainnumbers.image, 2);
idx_buenos = find(sigma ~= 0);
data = Trainnumbers.image(idx_buenos,:);
data_n = (data-media((idx_buenos)))./sigma(idx_buenos);

%% PCA
[coeff, data_pca, latent] = pca(data_n');

stem(latent)

save da