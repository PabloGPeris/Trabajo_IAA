%% Trabajo Inteligencia Artificial Aplicada
%% First Normalization
% Primera normalización, es decir, se calcula la media y la desviación
% típica y se normaliza con ellas. Se guardan los valores de la media y la
% desviación típica, para poder usarlas luego. Se eliminan los parámetros 
% con sigma 0, ya que no aportan información. Los datos que SÍ valen son
% los que tienen índice ind_validos.

clear
load Trainnumbers.mat

%% Normalización
% desv. típica y media de cada variable
sigma_total = std(Trainnumbers.image, 0, 2);
media_total = mean(Trainnumbers.image, 2);

% encuentra aquellos datos con sigma 0, que se eliminan
ind_validos = find(sigma_total ~= 0); 
data = Trainnumbers.image(ind_validos,:); % datos con  desv típica ~= 0

% media y sigma de los datos válidos
media_validos = media_total(ind_validos);
sigma_validos = sigma_total(ind_validos);

% datos normalizados
data_n = ((data - media_validos)./sigma_validos);

% guardado de datos
save datos_normalizacion data_n sigma_validos media_validos ind_validos