%% Trabajo Inteligencia Artificial Aplicada
%% TEST: Normalization and PCA
% Hace la normalización de los datos de test, así como la PCA (y el LDA si
% es necesario). - Este archivo dejará de existir

load datos_normalizacion % data_n sigma_validos media_validos ind_validos
load datos_PCA % coeff_pca D_pca data_r_pca

%% Datos de test
dtest = Trainnumbers.image(ind_validos,:); % cambiar por Testnumbers o lo que sea

%% Normalización
dtest_n = ((dtest - media_validos)./sigma_validos);

%% PCA - Probablemente hay que cambiar el númerode datos de la PCA y eso
dtest_r_pca = coeff_pca(:,1:D_pca)'*dtest_n;