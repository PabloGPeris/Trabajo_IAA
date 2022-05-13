%% matrix2digits
% Funcion para pasar de una matriz gigante de
% ceros y unos que sea la probabilidad de cada digito de ser uno u otro,
% pasarla a un array de esos digitos representados

function [digit_array] = matrix2digits(digit_matrix)
    
    sz = size(digit_matrix);
    digit_array = zeros(1, sz(2));

    for i = 1:sz(2)
        pos = find( digit_matrix(:,i) ~= 0 );
        digit_array(i) = pos-1;
    end
end