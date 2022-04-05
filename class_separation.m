function sep_data = class_separation(value, class)
%sep_data = CLASS_SEPARATION(value, class)
%
%   Separa las muestras value según su clase class. Devuelve un cell de
%   dimensión el número de clases.

% valores únicos de clase
C = unique(class);

% cell de los datos separados
sep_data = cell(1, length(C));

for i = 1:length(C)
    ind = class == C(i); % equivalente al find más o menos
    sep_data{i} = value(:, ind);
end

end