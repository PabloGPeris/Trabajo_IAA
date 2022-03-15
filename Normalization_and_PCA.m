%% Trabajo Inteligencia Artificial Aplicada
clear
load Trainnumbers.mat

%% Variables
% en tanto por uno, máximo error de reconstrucción que queremos poner
MSE_admisible = 0.1;

%% Normalizacióm
% desv. típica y media de cada variable
sigma_total = std(Trainnumbers.image, 0, 2);
media_total = mean(Trainnumbers.image, 2);

% encuentra aquellos datos con sigma 0, que se eliminan
ind_validos = find(sigma_total ~= 0); 
data = Trainnumbers.image(ind_validos,:);

% media y sigma de los datos válidos
media_validos = media_total(ind_validos);
sigma_validos = sigma_total(ind_validos);

% datos normalizados
data_n = ((data-media_validos)./sigma_validos);

save datos_normalizacion data_n sigma_validos media_validos
% hay que hacer función que normalice otros datos (los datos de test deben
% normalizarse rápido, así como PCAarse rápido)

%% PCA
% hace el PCA
% coeff = autovectores
% data_pca_inicial = datos en las dimensiones de la PCA (transpuesto)
% latent = autovalores = MSE esperado
[coeff, data_pca_inicial, latent] = pca(data_n'); 

% calcula el MSE esperado al quedarnos concada dimensión (en pu)
MSE_esperado = (sum(latent) - cumsum(latent))/sum(latent);

% número de dimensiones de PCA con las que nos quedamos según el MSE
% ademisible
D_pca = find(MSE_esperado <= MSE_admisible, 1)

% datos ya de dimensionalidad reducida
data_pca = data_pca_inicial(:, 1:D_pca)'; % Habría que irse planteando poner otros nombres

figure(31);
plot(0:length(latent), [1 MSE_esperado'], 'LineWidth', 1.5);
xline(D_pca, 'HandleVisibility','off')
yline(MSE_admisible, 'HandleVisibility','off')
xlabel('nº de dimensiones')
ylabel('MSE esperado (por unidad)')
xlim([0 length(latent)])
ylim([0 inf])

save datos_PCA coeff D_pca data_pca

%% Reconstrucción - lo meteré en alguna función 
% (completamente innecesario, pero queda bonito) - igual vale para ver si
% el número de dimensiones es suficiente
data_rec_n = (coeff(:, 1:D_pca)*data_pca);
data_rec = zeros(height(Trainnumbers.image), width(Trainnumbers.image));
data_rec(ind_validos, : ) = data_rec_n.*sigma_validos + media_validos;

%% Dibujos 
numero = [3 12 300 560]; % cambiar esto para ver magia

figure;
subplot(2,4,1)
digit_display(Trainnumbers.image, numero(1))
subplot(2,4,2)
digit_display(data_rec, numero(1))
subplot(2,4,3)
digit_display(Trainnumbers.image, numero(2))
subplot(2,4,4)
digit_display(data_rec, numero(2))
subplot(2,4,5)
digit_display(Trainnumbers.image, numero(3))
subplot(2,4,6)
digit_display(data_rec, numero(3))
subplot(2,4,7)
digit_display(Trainnumbers.image, numero(4))
subplot(2,4,8)
digit_display(data_rec, numero(4))


