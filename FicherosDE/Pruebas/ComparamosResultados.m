

clasificador1 = load ('Group1_dln.mat','class');
clasificador1 = clasificador1.class;
clasificador2 = load ('Group1_knn.mat','class');
clasificador2 = clasificador2.class;

%
N = length(clasificador1);

cont = 0; %Número de veces que no son iguales

posicion = ones(1,N);

for i = 1 : N
    if clasificador1(i) == clasificador2(i)
        %Estamos bien
    else
        cont = cont + 1;
        posicion(i) = 0; 
    end
end

cont

Pose = find(~posicion);



%% Prueba2

load label_man.mat

N = length(label_man);



clase_predicha = load ('Group1_dln.mat','class');


conf_mat = confusion(label_man, clase_predicha.class(:,1:N));
figure(3)
conf_chart = confusionchart(label_man, clase_predicha.class(:,1:N));



% Calcular accuracy
accuracy = trace(conf_chart.NormalizedValues/N);
disp(accuracy*100)






