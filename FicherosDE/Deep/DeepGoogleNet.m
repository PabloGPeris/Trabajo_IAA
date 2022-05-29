%% Cargamos los datos
addpath("..\")
load('trainedNetwork_4.mat');  

%% Prueba
load ("Test_prueba.mat")

N = length(Trainnumbers.image); % Sacamos dimensiones de los datos de test
%%

net = trainedNetwork_1;

for i = 1 : N

data_test = imread(sprintf('../../../ImagenesTest/FIG%d.jpeg',i));

class(1,i) = classify (net, data_test);

end

Group1_dln = double(class) - 1;

%%

%save('Group1_dln.mat',"Group1_dln")

