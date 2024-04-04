clear all;
warning('off');
sample_num = 10;
iter_nums = 100;
rng('default')
dataname = 'bc_pool_nci9.mat';
Dataset = load(dataname);
X = (Dataset.members);
Y = Dataset.gt;
nC = length(unique(Y));
[num_rows, num_cols] = size(X);

for i= 1:iter_nums
    seed = i;
    rng(seed);
    selected_columns = randperm(num_cols, sample_num);
    sample{1} = X(:, selected_columns);
    sample{2} = Y;
%%%%%%%%%%%%%%%%%%%%%%%SECTGL%%%%%%%%%%%%%%%%%%%%%%%Parameter = 0.6
    result_SECTGL = SECTGL(sample,0.6);
    final_result_SECTGL{i} = result_SECTGL;
end
%Res_Elements_SECTGL 
Res_Average{1} = {dataname,num2str(size(X,1)),'SECTGL',num2str(sum(cellfun(@(x) x(1), final_result_SECTGL))/iter_nums),num2str(sum(cellfun(@(x) x(2), final_result_SECTGL))/iter_nums),num2str(sum(cellfun(@(x) x(7), final_result_SECTGL))/iter_nums)};
disp("dataset:"+Res_Average{1}(1)+" sample_num:"+Res_Average{1}(2)+" ave_acc:"+Res_Average{1}(4) + " ave_nmi:"+Res_Average{1}(5)+ " ave_ari:"+Res_Average{1}(6))
