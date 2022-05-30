
%% Instrucciones
% Creamos una carpeta "ImagenesDeep" fuera de la carpeta del trabajo
% Dentro de esta carpeta creamos una carpeta llamada "Test" y otra llamada
% "Train" dentro de esta carpeta creamos 10 más que se llamen "0, 1, 2, ... 9"
% Luego ya podremos lanzar la sección siguiente


%% Guardado de las imagenes          

addpath("..\")
load('Test_numbers_HW1.mat')

image = Test_numbers.image;

N = length(image); 
x = 224; %dimension de las imagenes de entrada a la net


data_deepsaver2(image, x, N);


%save('datos_Deep.mat', 'Imagenes_convertidas', '-v7.3') % Me lo pide

%Comprobacion dimensiones guardado
%prueba = Imagenes_convertidas{2};

%% Prueba lectura

x = imread('../../ImagenesDeep/Train/0/FIG5.jpeg');

imshow(x);

size(x)