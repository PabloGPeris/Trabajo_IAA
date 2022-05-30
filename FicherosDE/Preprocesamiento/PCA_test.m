%% Trabajo Inteligencia Artificial Aplicada
%% PCA Test
% Primera PCA

clear
load datos_normalizacion_test.mat
load("..\..\datos_PCA.mat", "coeff_pca")
%% PCA

data_pca_test = (coeff_pca(:,1:end)'*data_n_test)';

save datos_PCA_test data_pca_test






