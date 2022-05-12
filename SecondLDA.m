

load datos_normalizacion.mat
load Trainnumbers.mat

rec = true; 
PCA = 9;
%%                         Falla exactamente lo mismo
%% LDA
sep_data = class_separation(data_n, Trainnumbers.label);

[SC, SW, SB, mC] = scatter_matrices(sep_data);
[W_lda,Diag] = eig(pinv(SW)*SB);

latent = real(diag(Diag));
W_lda = real(W_lda);
Diag = real(Diag);


%% Reconstrucción
% vale para ver si el número de dimensiones es suficiente
if rec  %#ok<*UNRCH>  
    
    N = width(Trainnumbers.image); % nº datos
    D_inicial = height(Trainnumbers.image); % nº dim iniciales

    %datos ya dimensionalidad reducida
    proyectados = W_lda(:,1:PCA)'*data_n;

    % datos reconstruidos normalizados
    data_rec_n = W_lda(:, 1:PCA)*proyectados;

    % datos reconstruidos desnormalizados
    data_rec = zeros(D_inicial, N);
    data_rec(ind_validos, : ) = data_rec_n.*sigma_validos + media_validos;
    
    rp = randperm(N, 4);
    figure(2);
    for i = 1:4
        subplot(2, 4, i*2 - 1)
        digit_display(Trainnumbers.image, rp(i))
        subplot(2, 4, i*2)
        digit_display(data_rec, rp(i))
    end
end