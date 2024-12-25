from matrix_class import Matrix
from datetime import datetime

input_matrix_line = 20
input_matrix_column = 30     # 输入矩阵、结果矩阵列数
hidden_layer = 20      # 隐藏层特征数
num_of_plies = 500    # 隐藏层数
num_of_term = 200  # 训练轮数
num_of_matrix = 100   # 样本矩阵数
lr = 1e-7          # 学习率
file_path = "data/goal_matrix.txt"
weight_matrix_file_path = "data/Weight_matrix.txt"
record_file_path = "data/record_matrix.txt"
if __name__ == '__main__':
    with open(weight_matrix_file_path, 'r+') as f:
        hidden = [[Matrix(input_matrix_column, hidden_layer), Matrix(hidden_layer, input_matrix_column)]
                  for i in range(num_of_plies)]    # 声明隐藏层
        # 读取参数矩阵
        weight_lines = f.readlines()
        for i in range(num_of_plies):
            for j in range(input_matrix_column):
                for k in range(hidden_layer):
                    hidden[i][0].set(j, k, float(weight_lines[i * input_matrix_column *
                                                              hidden_layer + j*hidden_layer+k].strip()))
            for j in range(hidden_layer):
                for k in range(input_matrix_column):
                    hidden[j][1].set(j, k, float(weight_lines[i * input_matrix_column * hidden_layer +
                                                              input_matrix_column * hidden_layer +
                                                              j*input_matrix_column+k].strip()))

    with (open(file_path, 'r+') as f):
        matrix_input = Matrix(input_matrix_line, input_matrix_column)
        matrix_output = Matrix(input_matrix_line, input_matrix_column)
        h = [Matrix(input_matrix_line, hidden_layer) for i in range(num_of_plies)]   # 隐藏层中间值

        lines = f.readlines()
        assert (input_matrix_line == int(lines[0])), "Value error! with input_matrix_line"
        assert (input_matrix_column == int(lines[1])), "Value error! with input_matrix_column"
        assert (num_of_matrix == int(lines[2])), "Value error! with num_of_matrix"

        for i in range(num_of_term):                 # 训练i次
            for j in range(num_of_matrix):            # 对于每个矩阵
                # 读取矩阵
                for line in range(input_matrix_line):
                    for column in range(input_matrix_column):
                        matrix_input.set(line, column, float(lines[line*input_matrix_column+column+3].strip()))
                        matrix_output.set(line, column, float(lines[line*input_matrix_column
                                                                    + column+3
                                                                    + input_matrix_column*input_matrix_line].strip()))

                # 计算所得矩阵
                for k in range(num_of_plies):
                    h[k] = matrix_input.mul(hidden[i][0])
                    matrix_input = h[k].mul(hidden[i][1])

                # 计算方差

                variance = matrix_input.variance(matrix_output)
                schedule = round((i*num_of_matrix+j)*100/(num_of_term * num_of_matrix), 2)

                # 进行记录
                print(f"这是第{i}次训练中的第{j}个矩阵，所得到方差结果为{variance}，当前进度{schedule}%")
                with open(record_file_path, 'a') as f1:
                    f1.write(str(datetime.utcnow()))
                    f1.write(f"这是第{i}次训练中的第{j}个矩阵，所得到方差结果为{variance}，当前进度{schedule}%\n")
                    f1.write("以下为结果矩阵")
                    for line in range(input_matrix_line):
                        for column in range(input_matrix_column):
                            f1.write(str(matrix_input.matrix[line][column])+"\t\t\t")
                        f1.write("\n")
                    f1.write("以下为目标矩阵")
                    for line in range(input_matrix_line):
                        for column in range(input_matrix_column):
                            f1.write(str(matrix_output.matrix[line][column])+"\t\t\t")
                        f1.write("\n")
                    f1.write("\n\n")

                # 计算结果矩阵梯度
                grad_output = matrix_output.sub(matrix_input)
                grad_hidden = [Matrix(hidden_layer, input_matrix_column)
                               for _ in range(num_of_plies)]
                grad_input = grad_output

                # 计算权重矩阵梯度
                for k in reversed(range(num_of_plies)):
                    grad_hidden[k] = grad_input.mul(hidden[k][1].transpose())
                    grad_input = grad_hidden[k].mul(hidden[k][0].transpose())
                    # 更新权重
                    hidden[k][1] = hidden[k][1].sub(h[k].transpose().mul(grad_output).mul_num(lr))
                    hidden[k][0] = hidden[k][0].sub(matrix_input.transpose().mul(grad_hidden[k]).mul_num(lr))

                # 更新文档中权重
                with open(weight_matrix_file_path, 'r+') as f1:
                    f1.truncate(0)
                    for n in range(num_of_plies):
                        for m in range(input_matrix_column):
                            for k in range(hidden_layer):
                                f1.write(f"{hidden[n][0].matrix[m][k]}\n")
                        for m in range(hidden_layer):
                            for k in range(input_matrix_column):
                                f1.write(f"{hidden[n][1].matrix[m][k]}\n")
        print("训练完成")
