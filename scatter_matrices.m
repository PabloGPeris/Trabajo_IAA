function [SC, SW, SB, mC] = scatter_matrices(sep_data)

nC = length(sep_data);

SC = cell(1, nC);
mC = cell(1, nC);
SB = 0;
SW = 0;
msum = 0;

for i = 1:nC
    mC{i} = mean(sep_data{i}, 2);
    %msum = msum + mC{i};    
    SC{i} = (sep_data{i}-mC{i})*(sep_data{i}-mC{i})';
    SW = SW + SC{i};
    SB = SB + length(sep_data{i})*mC{i}*mC{i}';
end

%m = msum/nC;
% SB1 = 0;
% for i = 1:nC      %Da exactamente igual a SB
%     SB1 = SB1 + length(sep_data{i})*(mC{i}-m)*(mC{i}-m)';
% end
end
