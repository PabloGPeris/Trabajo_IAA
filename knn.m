
%load Testnumbers.mat
load Trainnumbers.mat
load datos_PCA.mat
%load datos_PCA_2.mat
%load datos_lda.mat
%% Datos a pelo
knnMdl = fitcknn(Trainnumbers.image',Trainnumbers.label');         %Proceso aprendizaje

knnclassp = predict(knnMdl,Trainnumbers.image'); 
%knnclasst = predict(knnMdl,Testnumbers.image');         %Cual cree que son de cada clase

no_errors_nn_p = length(find(knnclassp'~=Trainnumbers.label)) %0 ya que memoriza los datos train
%no_errors_nn_t = length(find(knnclasst'~=Testnumbers.label))     %El numero que se ha equivocado de los datos test



%% Datos PCA1
%pca de los datos del test = datos_pca_test
knnMdl_PCA1 = fitcknn(data_pca',Trainnumbers.label');         

knnclassp_PCA1 = predict(knnMdl_PCA1,data_pca'); 
%knnclasst_PCA1 = predict(knnMdl_PCA1,datos_pca_test');        

no_errors_nn_p_PCA1 = length(find(knnclassp_PCA1'~=Trainnumbers.label)) 
%no_errors_nn_t_PCA1 = length(find(knnclasst_PCA1'~=Testnumbers.label))

%% Datos PCA2
%pca de los datos del test = datos_pca_test_2
knnMdl_PCA2 = fitcknn(data_pca_2',Trainnumbers.label');         

knnclassp_PCA2 = predict(knnMdl_PCA2,data_pca_2');  
knnclasst_PCA2 = predict(knnMdl_PCA2,datos_pca_test_2');        

no_errors_nn_p_PCA2 = length(find(knnclassp_PCA2'~=Trainnumbers.label)) 
no_errors_nn_t_PCA2 = length(find(knnclasst_PCA2'~=Testnumbers.label))

%% Datos LDA
%lda de los datos del test = datos_lda_test
knnMdl_LDA = fitcknn(data_lda',Trainnumbers.label');         

knnclassp_LDA = predict(knnMdl_LDA,data_lda'); 
%knnclasst_LDA = predict(knnMdl_LDA,datos_lda_test');        

no_errors_nn_p_LDA = length(find(knnclassp_LDA'~=Trainnumbers.label)) 
%no_errors_nn_t_LDA = length(find(knnclasst_LDA'~=Testnumbers.label))

%% Gráficas

Y = [no_errors_nn_t             %Sin los datos de test no podemos evaluar
    no_errors_nn_t_PCA1
    no_errors_nn_t_PCA2
    no_errors_nn_t_LDA
     ];
figure
bar(Y)
title('Errores datos test de los respectivos knn')



