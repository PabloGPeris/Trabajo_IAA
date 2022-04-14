%% Carga de datos ya normalizados
%clear
load datos_normalizacion.mat
%load datos_normalizacion_test.mat
%load Testnumbers.mat
load Trainnumbers.mat

[D,N]=size(Trainnumbers.image); 



%% Propuesta: cambiar el número de clases K-means??



%% Variables

D_lda = 673;    %Dimensiones que queremos para el LDA (673,9)Límites 


%% LDA
sep_data = class_separation(data_n, Trainnumbers.label);
[SC, SW, SB, mC] = scatter_matrices(sep_data);

[W,Diag] = eig(inv(SW)*SB);

%%

proyectados_train = W(:,D_lda)'*data_n;
reconstruidos_train = W(:,D_lda)*proyectados_train;

%proyectados_test = W(:,D_lda)'*data_n_test;
%reconstruidos_test = W(:,D_lda)*proyectados_test;

%% Desnormalización de los datos proyectados

for i=1:N
    reconstruidos_train_des(:,i)=(reconstruidos_train(:,i).*sigma_validos) + media_validos;
end
% for i=1:N
%     reconstruidos_test_des(:,i)=(reconstruidos_test(:,i).*sigma_validos) + media_validos;
% end 
% proyectados_test_des = (proyectados_train.*sigma_validos) + media_validos;

% MSE = D*mse(Testnumbers.image-reconstruidos_test_des);

%%
%sep_data = class_separation(valor, class)                                             %Como funciona esto??


% Queremos saber la longitud de las clases 
% x1t -> numero de etiquetas de la clase 1
% 
% figure;
% plot(reconstruidos_test_des(1,:),reconstruidos_test_des(2,:),'b*')
% hold on;
% plot(proyectados_test_des(1,:),proyectados_test_des(2,:),'g*')  
% hold on;
%plot(Testnumbers.image(1,x0t),Testnumbers.image(2,x0t),'r*'); 
% hold on;
% plot(Testnumbers.image(1,x1t),Testnumbers.image(2,x1t),'r*'); 
% hold on;
% plot(Testnumbers.image(1,x2t),Testnumbers.image(2,x2t),'k*');
% title('Datos test desnormalizados');





%% Figuras 
numero = 12; % cambiar esto para ver magia

figure;
subplot(1,2,1)
digit_display(Trainnumbers.image, numero)
subplot(1,2,2)
digit_display(data_rec, numero)





















