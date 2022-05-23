%% Guardado de las imagenes          

% addpath("..\")
% load Trainnumbers.mat
% 
% % Datos
% 
% image = Trainnumbers.image;
% label = Trainnumbers.label;
% 
% % tanto por uno de datos que se usan para entrenar (no para test)
% PD = 0.8;
% N = length(label); 
% 
% x = 32; %dimension de las imagenes de entrada a la net
% 
% 
% 
% data_deepsaver_lenet(image, label, x, N, PD);

%%
x = imread('../../ImagenesDeep/lenet/Test/FIG1.jpeg');

imshow(x);

size(x)

%%

save('trainedNetworklenet','trainedNetwork_1')
save('trainInfolenet','trainInfoStruct_1')

%%
label_test = load ('../../ImagenesDeep/lenet/label_test.mat');

load('trainedNetworklenet.mat');
net = trainedNetwork_1;

%deepNetworkDesigner(net)

%analyzeNetwork(net);

for i = 1 : 2000

data_test = imread(sprintf('../../ImagenesDeep/lenet/Test/FIG%d.jpeg',i));

class(1,i) = classify (net, data_test);

end

clase = double(class) - 1;

accuracy = sum(clase == label_test.label_test) /numel(label_test.label_test);

disp ("accuracy: ");
disp(accuracy*100);

% Confussion matrix

C = confusionmat(label_test.label_test,clase);

confusionchart(C);

%% 

load('trainInfolenet.mat');





