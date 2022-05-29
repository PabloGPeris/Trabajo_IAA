%% Guardado de las imagenes          

addpath("..\")
load Trainnumbers.mat

% Datos

image = Trainnumbers.image;
label = Trainnumbers.label;

% tanto por uno de datos que se usan para entrenar (no para test)
PD = 1;
N = length(label); 

x = 32; %dimension de las imagenes de entrada a la net



data_deepsaver_lenet(image, label, x, N, PD);

%%
x = imread('../../ImagenesDeep/lenet/Test/FIG1.jpeg');

imshow(x);

size(x)
