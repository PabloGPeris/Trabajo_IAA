clear

addpath("..\")

load Trainnumbers.mat % para la clasificación básicamente
load("datos_PCA.mat", "data_pca"); % sin hacer PCA previa

%%
sep_data = class_separation(data_PCA, Trainnumbers.label)