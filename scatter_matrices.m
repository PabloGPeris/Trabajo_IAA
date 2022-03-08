function [SC, SW, SB, mC] = scatter_matrices(sep_data)

nC = length(sep_data);

SC = cell(1, nC);
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
