function [SC, SW, SB, mC] = scatter_matrices(sep_data)
%[SC, SW, SB, mC] = SCATTER_MATRICES(sep_data)
%
%   Calcula las matrices de dispersión. SC = {S1, S2, ..., S(#grupos)} es
%   las matrices de dispersión de grupo; SW es la matriz de dispersión
%   dentro de los grupos; SB, entre los grupos. mC es la media de cada
%   grupo. 
%   Importante: Los grupos se suponene normalizados.
nC = length(sep_data);

SC = cell(1, nC);
% mC = media of Clusters
mC = cell(1, nC);
SB = 0;
SW = 0;
for i = 1:nC
    mC{i} = mean(sep_data{i}, 2);
    SC{i} = (sep_data{i}-mC{i})*(sep_data{i}-mC{i})';
    SW = SW + SC{i};
    SB = SB + length(sep_data{i})*mC{i}*mC{i}';
end

end
