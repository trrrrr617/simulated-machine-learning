from matrix_class import Matrix
from datetime import datetime

input_layer = 20
output_layer = 20     # 输入、结果特征数
hidden_layer = 20      # 隐藏层特征数
num_of_plies = 1    # 隐藏层层数
num_of_term = 5000  # 训练轮数
num_of_sample = 1   # 样本个数
lr = (1e-3)*1          # 学习率
file_path = "data/goal_vector.txt"
weight_matrix_file_path = "data/Weight_vector.txt"
record_file_path = "data/record_vector.txt"
if __name__ == '__main__':
    average = 0
    with open(weight_matrix_file_path, 'r+') as f:
        hidden_weight = [Matrix(hidden_layer, hidden_layer) for i in range(num_of_plies-1)]    # 声明隐藏层权重矩阵
        hidden_first_weight = Matrix(input_layer, hidden_layer)
        hidden_last_weight = Matrix(hidden_layer, output_layer)
        # 读取参数矩阵
        weight_lines = f.readlines()
        assert (input_layer == int(weight_lines[0])), "Value error! with input_layer == int(weight_lines[0])"
        assert (hidden_layer == int(weight_lines[1])), "Value error! with hidden_layer == int(weight_lines[1])"
        assert (num_of_plies == int(weight_lines[3])), "Value error! with num_of_plies == int(weight_lines[3])"
        assert (output_layer == int(weight_lines[2])), "Value error! with output_layer == int(weight_lines[2])"
        for j in range(input_layer):
            for k in range(hidden_layer):
                hidden_first_weight.set(j, k, float(weight_lines[j*hidden_layer+k+4].strip()))
        for i in range(num_of_plies-1):
            for j in range(hidden_layer):
                for k in range(hidden_layer):
                    hidden_weight[i].set(j, k, float(weight_lines[input_layer*hidden_layer +
                                                                  k + j*hidden_layer +
                                                                  i*hidden_layer*hidden_layer+4].strip()))
        for j in range(hidden_layer):
            for k in range(output_layer):
                hidden_last_weight.set(j, k, float(weight_lines[input_layer*hidden_layer +
                                                                (num_of_plies-1)*hidden_layer*hidden_layer +
                                                                k + j*output_layer+4].strip()))

    # 打开存有原始值和目标值的文件
    with (open(file_path, 'r+') as f):
        matrix_input = Matrix(1, input_layer)
        matrix_output = Matrix(1, output_layer)
        matrix_goal = Matrix(1, output_layer)
        h = [Matrix(1, hidden_layer) for i in range(num_of_plies)]   # 隐藏层中间值

        lines = f.readlines()
        assert (input_layer == int(lines[0])), "Value error! with input_matrix_line"
        assert (output_layer == int(lines[1])), "Value error! with input_matrix_column"
        assert (num_of_sample == int(lines[2])), "Value error! with num_of_matrix"

        for i in range(num_of_term):                 # 训练i次
            for j in range(num_of_sample):            # 对于每个矩阵
                # 读取矩阵
                for num in range(input_layer):
                    matrix_input.set(0, num, float(lines[j*input_layer + j*output_layer + num+3].strip()))
                for num in range(output_layer):
                    matrix_goal.set(0, num, float(lines[j*output_layer + (j+1)*input_layer + num+3].strip()))

                # 计算所得矩阵
                h[0] = matrix_input.mul(hidden_first_weight)
                for k in range(num_of_plies-1):
                    h[k+1] = h[k].mul(hidden_weight[k])
                matrix_output = h[num_of_plies-1].mul(hidden_last_weight)

                # 计算方差
                variance = matrix_output.variance(matrix_goal)
                schedule = round((i*num_of_sample+j)*100/(num_of_term * num_of_sample), 2)
                average = (average*(i*num_of_sample+j) + variance)/(i*num_of_sample+j+1)
                # 进行记录
                print(f"这是第{i}次训练中的第{j}个矩阵，所得到方差结果为{variance}，当前进度{schedule}%，当前平均方差{average}")
                with open(record_file_path, 'a') as f1:
                    f1.write(str(datetime.utcnow()))
                    f1.write(f"这是第{i}次训练中的第{j}个向量，所得到方差结果为{variance}，当前进度{schedule}%，当前平均方差{average}\n")
                    f1.write("以下为输入向量")
                    for num in range(input_layer):
                        f1.write(str(matrix_input.matrix[0][num])+"\t\t\t")
                    f1.write("\n")
                    f1.write("以下为结果向量")
                    for num in range(output_layer):
                        f1.write(str(matrix_output.matrix[0][num])+"\t\t\t")
                    f1.write("\n")
                    f1.write("以下为目标向量")
                    for num in range(output_layer):
                        f1.write(str(matrix_goal.matrix[0][num])+"\t\t\t")
                    f1.write("\n")
                    f1.write("\n\n")

                # 计算结果矩阵梯度
                grad_output = matrix_goal.sub(matrix_output)
                grad_hidden_last = h[num_of_plies-1].transpose().mul(grad_output)
                grad_hidden = [Matrix(hidden_layer, hidden_layer)
                               for _ in range(num_of_plies-1)]
                grad = grad_output.mul(grad_hidden_last.transpose())
                for k in reversed(range(num_of_plies-2)):
                    grad_hidden[k] = h[k].transpose().mul(grad)
                    grad = grad.mul(hidden_weight[k].transpose())
                grad_hidden_first = matrix_input.transpose().mul(grad)

                # 更新权重矩阵
                hidden_last_weight = hidden_last_weight.plus(grad_hidden_last.mul_num(lr))
                for k in range(num_of_plies - 1):
                    hidden_weight[k] = hidden_weight[k].plus(grad_hidden[k].mul_num(lr))
                hidden_first_weight = hidden_first_weight.plus(grad_hidden_first.mul_num(lr))

                # 更新文档中权重
                with open(weight_matrix_file_path, 'r+') as f1:
                    f1.seek(0)
                    f1.truncate(0)
                    f1.writelines(str(input_layer)+"\n")
                    f1.writelines(str(hidden_layer)+"\n")
                    f1.writelines(str(output_layer)+"\n")
                    f1.writelines(str(num_of_plies)+"\n")
                    for m in range(input_layer):
                        for n in range(hidden_layer):
                            f1.write(f"{hidden_first_weight.matrix[m][n]}\n")
                    for k in range(num_of_plies-1):
                        for m in range(hidden_layer):
                            for n in range(hidden_layer):
                                f1.write(f"{hidden_weight[k].matrix[m][n]}\n")
                    for m in range(hidden_layer):
                        for n in range(output_layer):
                            f1.write(f"{hidden_last_weight.matrix[m][n]}\n")

        print("训练完成")
