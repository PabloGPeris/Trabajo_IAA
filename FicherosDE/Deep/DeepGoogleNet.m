%% Cargamos los datos
addpath("..\")
load('trainedNetwork_4.mat');  

%% Prueba
load('Test_numbers_HW1.mat')

N = length(Test_numbers.image); % Sacamos dimensiones de los datos de test
%%

net = trainedNetwork_1;

for i = 1 : N

data_test = imread(sprintf('../../../ImagenesTest/FIG%d.jpeg',i));

clase(1,i) = classify (net, data_test);

end

class = double(clase) - 1;

%%
class = [0 1 0 1 0 1];
name = {'LuisBade', 'PabloPer', 'JaviDiaz'};
PCA = 0;
%save('Group1_dln.mat', "name", "PCA", "class")

