function [data_lda, D_lda_max] = KmeansLDAfunction(k)

addpath("..\")
load Trainnumbers.mat % para la clasificación básicamente
load datos_normalizacion.mat
[D,N]=size(Trainnumbers.image); % nº dim iniciales   nº datos


[new_data, new_label] = clustering_kmeans(data_n, Trainnumbers.label, k);


sep_data = class_separation(new_data, new_label);

[SC, SW, SB, mC] = scatter_matrices(sep_data);
[W_lda,Diag] = eig(pinv(SW)*SB);

D_lda_max = 10*k-1;    %Dimensiones que queremos para el LDA (10*K-1 máximo)

%datos ya dimensionalidad reducida
proyectados = W_lda(:,1:D_lda_max)'*new_data; 
% datos reconstruidos normalizados
reconstruidos = W_lda(:,1:D_lda_max)*proyectados;   

data_lda = reconstruidos';


end