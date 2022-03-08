function sep_data = class_separation(valor, class)
% valores en forma 

C = unique(class);
sep_data = cell(1, length(C));

for i = 1:length(C)
    indices = class == C(i);
    sep_data{i} = valor(:, indices);
end

end