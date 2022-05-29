
load Group1_knn.mat
load Group1_dln.mat

clasificador1 = Group1_knn;
clasificador2 = Group1_dln;

N = length(clasificador1);

cont = 0; %Número de veces que no son iguales

posicion = zeros(1,N);

for i = 1 : N
    if clasificador1(i) == clasificador2(i)
        %Estamos bien
    else
        cont = cont + 1;
        posicion(i) = i; 
    end
end

Pose = find(~0);
