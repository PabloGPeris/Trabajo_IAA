function [new_data, new_label] = clustering_kmeans_v2(data, label, k)
%[new_data, new_label] = CLUSTERING_KMEANS(data, label, k)
%   
%   Mediante k-means, construye k nuevos clustéres de cada clase (label) de
%   los datos de entrada.

C = unique(label);

% separar los datos por clase
nc = length(C); % nº clases
nd = length(label);

new_label = zeros(1, nd);

for i = 1:nc
    ind = label == C(i); 

    % k-means para cada grupo
    try
        ind2 = kmeans(data(:,ind)', k, 'replicates', 5)'; 
    catch ME
        disp(ME)
    end

    % nuevas etiquetas (sumarle nc a los nuevos subgrupos)
    new_label(ind) = nc*(ind2-1) + C(i);

    % nuevos datos
end

new_data = data;
end