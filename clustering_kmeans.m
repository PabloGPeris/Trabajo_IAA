function [new_data, new_label] = clustering_kmeans(data, label, k)
%[new_data, new_label] = CLUSTERING_KMEANS(data, label, k)
%   
%   Mediante k-means, construye k nuevos clustéres de cada clase (label) de
%   los datos de entrada.

% separar los datos por clase
sep_data = class_separation(data, label);

new_label = [];
new_data = [];

nc = length(sep_data); % nº clases

for i = 0:nc-1
    % k-means para cada grupo
    try
        ind = kmeans(sep_data{i + 1}', k); 
    catch ME
        disp 'patata'
    end


    
    % nuevas etiquetas (sumarle nc a los nuevos subgrupos)
    new_label_i = zeros(1, length(sep_data{i + 1}));
    for j = 1:k
        new_label_i(ind == j) = nc*(j-1) + i;
    end

    % nuevos datos
    new_label = [new_label new_label_i];
    new_data = [new_data sep_data{i + 1}];
end
end