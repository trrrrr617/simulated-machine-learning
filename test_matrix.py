from matrix_class import Matrix
input_matrix_line = 20
input_matrix_column = 30     # 输入矩阵、结果矩阵列数
hidden_layer = 20      # 隐藏层特征数
num_of_plies = 500    # 隐藏层数
file_path = "data/test_matrix.txt"
weight_matrix_file_path = "data/Weight_matrix.txt"
if __name__ == '__main__':
    with open(weight_matrix_file_path, 'r+') as f:
        # 声明隐藏层
        hidden = [[Matrix(input_matrix_column, hidden_layer), Matrix(hidden_layer, input_matrix_column)]
                  for i in range(num_of_plies)]
        # 读取参数矩阵
        weight_lines = f.readlines()
        for i in range(num_of_plies):
            for j in range(input_matrix_column):
                for k in range(hidden_layer):
                    hidden[i][0].set(j, k, float(weight_lines[j*hidden_layer+k].strip()))
            for j in range(hidden_layer):
                for k in range(input_matrix_column):
                    hidden[j][1].set(j, k, float(weight_lines[j*input_matrix_column+k].strip()))

    with (open(file_path, 'r+') as f):
        matrix_input = Matrix(input_matrix_line, input_matrix_column)
        matrix_output = matrix_input.mul_num(1/2)
        h = Matrix(input_matrix_line, hidden_layer)   # 隐藏层中间值

        lines = f.readlines()
        assert (input_matrix_line == int(lines[0])), "Value error! with input_matrix_line"
        assert (input_matrix_column == int(lines[1])), "Value error! with input_matrix_column"
        lines = lines[2:]

        line_num = 0
        for line in lines:
            numbers = line.strip().split()
            numbers = [float(num) for num in numbers]
            for num in range(input_matrix_column):
                matrix_input.set(line_num, num, numbers[num])
            line_num = line_num + 1

        # 计算所得矩阵
        for k in range(num_of_plies):
            h = matrix_input.mul(hidden[i][0])
            matrix_input = h.mul(hidden[i][1])
        print(f"所得方差为{matrix_input.variance(matrix_output)}")

        # 写入矩阵
        f.write("\n\n\n"+"以下为结果矩阵\n\n")
        for i in range(input_matrix_line):
            for j in range(input_matrix_column):
                f.write(str(matrix_input.matrix[i][j])+"\t\t\t")
            f.write("\n")
        print("写入所得矩阵完毕")
