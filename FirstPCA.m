%% Trabajo Inteligencia Artificial Aplicada
%% First PCA
% Primera PCA que se hace, donde se obtienen los coeficientes y todas esas
% cosas de la PCA, guardando tanto los datos como ... PATATA

clear
load datos_normalizacion.mat

%% Variables
% en tanto por uno, máximo error de reconstrucción que queremos poner
MSE_admisible = 0.5;

%% PCA
% hace el PCA
% coeff_pca = autovectores
% data_pca = datos en las dimensiones de la PCA (transpuesto)
% latent = autovalores = MSE esperado
[coeff_pca, data_pca, latent] = pca(data_n'); 

% calcula el MSE esperado al quedarnos concada dimensión (en pu) - La suma
% de latent es, evidente, el número de dimensiones total = 673, pero queda
% más claro así puesto
MSE_esperado = (sum(latent) - cumsum(latent))/sum(latent);

% número de dimensiones de PCA con las que nos quedamos según el MSE
% admisible
D_pca = find(MSE_esperado <= MSE_admisible, 1)

% datos ya de dimensionalidad reducida
data_pca_r = data_pca(:, 1:D_pca)';

% otra forma de obtener los datos de dimensionalidad reducida sería:
% data_pca_r = coeff(:,1:D_pca)'*data_n ;

figure(31);
plot(0:length(latent), [1 MSE_esperado'], 'LineWidth', 1.5);
xline(D_pca, 'HandleVisibility','off')
yline(MSE_admisible, 'HandleVisibility','off')
xlabel('nº de dimensiones')
ylabel('MSE esperado (por unidad)')
xlim([0 length(latent)])
ylim([0 inf])

save datos_PCA coeff_pca D_pca data_pca_r