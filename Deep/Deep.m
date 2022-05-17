%% Guardado de las imagenes          

addpath("..\")
load Trainnumbers.mat

% Datos

image = Trainnumbers.image;
label = Trainnumbers.label;

% tanto por uno de datos que se usan para entrenar (no para test)
PD = 0.8;
N = length(label); 
x = 224; %dimension de las imagenes de entrada a la net



data_deepsaver2(image, label, x, N, PD);


%save('datos_Deep.mat', 'Imagenes_convertidas', '-v7.3') % Me lo pide

%Comprobacion dimensiones guardado
%prueba = Imagenes_convertidas{2};


%%

x = imread('../../ImagenesDeep/Test/FIG1.jpeg');

imshow(x);

size(x)

                                    %% Empezamos con el DEEP
%% Modificación de GoogleNet
net = googlenet;

%inputSize = net.Layers(1).InputSize

%analyzeNetwork(net);

deepNetworkDesigner(net)    %Modificamos la net ya hecha         


%net = trainNetwork (imagesTrain, layers, options);
%% Guardamos la red entrenada



save('trainedNetwork_2','trainedNetwork_1')

%% Testeo de la net

label_test = load ('../../ImagenesDeep/label_test.mat');

load('trainedNetwork_2.mat');  % La net 1 tienes unos datos de entrenamiento diferentes
net = trainedNetwork_1;

for i = 1 : 2000

data_test = imread(sprintf('../../ImagenesDeep/Test/FIG%d.jpeg',i));

class(1,i) = classify (net, data_test);

end

clase = double(class) - 1;

accuracy = sum(clase == label_test.label_test) /numel(label_test.label_test);

disp ("accuracy: ");
disp(accuracy*100);

% Confussion matrix

C = confusionmat(label_test.label_test,clase);

confusionchart(C);


