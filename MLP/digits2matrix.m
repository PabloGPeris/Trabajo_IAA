%% digits2matrix
% Funcion para pasar de un array de digitos a una matriz gigante de
% ceros y unos que sea la probabilidad de cada digito de ser uno u otro

function [digit_matrix] = digits2matrix(digit_array)
    
    sz = size(digit_array);
    digit_matrix = zeros(10, sz(2));

    for i = 1:sz(2)
        number = digit_array(i);
        digit_matrix(number + 1, i) = 1;
    end
end