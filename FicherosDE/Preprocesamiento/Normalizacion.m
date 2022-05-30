
%Normalizacion datos de train
load Trainnumbers.mat

sigma_total = std(Trainnumbers.image, 0, 2);
media_total = mean(Trainnumbers.image, 2);                  
                                                               
% encuentra aquellos datos con sigma 0, que se eliminan
ind_validos = find(sigma_total ~= 0); 
%data = Trainnumbers.image(ind_validos,:);

% media y sigma de los datos válidos
media_validos = media_total(ind_validos);
sigma_validos = sigma_total(ind_validos);

% datos normalizados
%data_n = ((data-media_validos)./sigma_validos);

%save datos_normalizacion data_n sigma_validos media_validos sigma_validos ind_validos

%% Lo dejamos comentado hasta tener los datos de test
%Normalización de datos de test
load Test_numbers_HW1.mat        %Suponemos que los datos de test se llaman así
data_test = Test_numbers.image(ind_validos,:);
data_n_test = ((data_test-media_validos)./sigma_validos);

save datos_normalizacion_test data_n_test ind_validos


%% Tengo dudas pero creo que esto

load Test_numbers_HW1.mat

sigma_total = std(Test_numbers.image, 0, 2);
media_total = mean(Test_numbers.image, 2);

ind_validos = find(sigma_total ~= 0); 

media_validos = media_total(ind_validos);
sigma_validos = sigma_total(ind_validos);

data_test = Test_numbers.image(ind_validos,:);
data_n_test = ((data_test-media_validos)./sigma_validos);

save datos_normalizacion_test data_n_test ind_validos
