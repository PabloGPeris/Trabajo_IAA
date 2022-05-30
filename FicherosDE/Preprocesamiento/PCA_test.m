
clear
load datos_normalizacion_test.mat


[coeff_pca_test, data_pca_test, latent_test] = pca(data_n_test');


save datos_PCA_test data_pca_test coeff_pca_test latent_test






