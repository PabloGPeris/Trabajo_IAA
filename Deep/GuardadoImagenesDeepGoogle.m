
%% Instrucciones
% Creamos una carpeta "ImagenesDeep" fuera de la carpeta del trabajo
% Dentro de esta carpeta creamos una carpeta llamada "Test" y otra llamada
% "Train" dentro de esta carpeta creamos 10 más que se llamen "0, 1, 2, ... 9"
% Luego ya podremos lanzar la sección siguiente


%% Guardado de las imagenes          

addpath("..\")
load Trainnumbers.mat

% Datos

image = Trainnumbers.image;
label = Trainnumbers.label;

% tanto por uno de datos que se usan para entrenar (no para test)
PD = 1;
N = length(label); 
x = 224; %dimension de las imagenes de entrada a la net



data_deepsaver2(image, label, x, N, PD);


%save('datos_Deep.mat', 'Imagenes_convertidas', '-v7.3') % Me lo pide

%Comprobacion dimensiones guardado
%prueba = Imagenes_convertidas{2};

%% Prueba lectura

x = imread('../../ImagenesDeep/Train/0/FIG5.jpeg');

imshow(x);

size(x)