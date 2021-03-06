%% Carga de datos ya normalizados
clear all
load Trainnumbers.mat
load datos_normalizacion.mat
[D,N]=size(Trainnumbers.image); % nº dim iniciales   nº datos


%% LDA
sep_data = class_separation(data_n, Trainnumbers.label);

[SC, SW, SB, mC] = scatter_matrices(sep_data);
[W_lda,Diag] = eig(pinv(SW)*SB);

latent = real(diag(Diag));
W_lda = real(W_lda);
Diag = real(Diag);

%% Para observar la evolución del MSE en función de las dimensiones
% Hasta que dimesiones queremos testear
Num_dim = 1; 

num_clases = 10; %nº de clases de los datos
dif = num_clases-Num_dim;
MSE = zeros(1,dif);

for D_lda = num_clases-1:-1:Num_dim   

    % LDA
   
    %datos ya dimensionalidad reducida
    proyectados = W_lda(:,1:D_lda)'*data_n;
    % datos reconstruidos normalizados
    reconstruidos = W_lda(:,1:D_lda)*proyectados;

    reconstruidos_des = zeros(D,N);
    % datos reconstruidos desnormalizados
    reconstruidos_des(ind_validos, : ) = (reconstruidos.*sigma_validos) + media_validos;
    % reconstruidos_des = real(reconstruidos_des);

    % Error
    MSE(dif) = D*mse(Trainnumbers.image-reconstruidos_des);

    dif = dif - 1;

end

%Ploteamos evolución del error MSE en función del numero de dimensiones que le metemos    Diría que es algo inútil, pero ahí lo tenemos  
figure;
plot(num_clases-1:-1:Num_dim, MSE(1,end:-1:1), 'LineWidth', 1.5); 
%plot(num_clases:-1:Num_dim, MSE(num_clases:-1:Num_dim), 'LineWidth', 1.5);  
%plot(Num_dim:num_clases, MSE(Num_dim:num_clases), 'LineWidth', 1.5);
xlabel('nº de dimensiones')
ylabel('MSE (por unidad)')


%% Variables

D_lda = 9;    %Dimensiones que queremos para el LDA (9 máximo)


%%
%datos ya dimensionalidad reducida
proyectados = W_lda(:,1:D_lda)'*data_n; %=data_r_pca
% datos reconstruidos normalizados
reconstruidos = W_lda(:,1:D_lda)*proyectados;   %=data_rec_n


reconstruidos_des = zeros(D,N);
% datos reconstruidos desnormalizados
reconstruidos_des(ind_validos, : ) = (reconstruidos.*sigma_validos) + media_validos;    %=data_rec

rp = randperm(N, 4);
figure;
for i = 1:4
    subplot(2, 4, i*2 - 1)
    digit_display(Trainnumbers.image, rp(i))
    subplot(2, 4, i*2)
    digit_display(reconstruidos_des, rp(i))    %Los valores reconstruidos dan valores muy raros, 
                                                    %siempre muestra los
                                                    %mismo con pequeñas
                                                    %diferencias
end


% Error
MSE = D*mse(Trainnumbers.image-reconstruidos_des); 

%pasamos a nomenclatura Pablo xddd
data_lda = reconstruidos';
coeff_lda = W_lda;
data_r_lda = proyectados;
latent_lda = latent;
%save datos_LDA data_lda coeff_lda latent_lda
%%
num = 30;
figure(1);
subplot(1, 2, 1)
digit_display(Trainnumbers.image, num)
subplot(1, 2, 2)
digit_display(reconstruidos_des, num)














