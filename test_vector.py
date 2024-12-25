from matrix_class import Matrix

input_layer = 20
output_layer = 20     # 输入、结果特征数
hidden_layer = 20      # 隐藏层特征数
num_of_plies = 1    # 隐藏层层数
lr = 1e-7          # 学习率
file_path = "data/test_vector.txt"
weight_matrix_file_path = "data/Weight_vector.txt"
record_file_path = "data/record_vector.txt"
if __name__ == '__main__':
    with open(weight_matrix_file_path, 'r+') as f:
        hidden_first_weight = Matrix(input_layer, hidden_layer)
        hidden_last_weight = Matrix(hidden_layer, output_layer)
        hidden_weight = [Matrix(hidden_layer, hidden_layer) for i in range(num_of_plies-1)]    # 声明隐藏层权重矩阵
        # 读取参数矩阵
        weight_lines = f.readlines()
        assert (input_layer == int(weight_lines[0])), "Value error! with input_layer == int(weight_lines[0])"
        assert (hidden_layer == int(weight_lines[1])), "Value error! with hidden_layer == int(weight_lines[1])"
        assert (output_layer == int(weight_lines[2])), "Value error! with output_layer == int(weight_lines[2])"
        assert (num_of_plies == int(weight_lines[3])), "Value error! with num_of_plies == int(weight_lines[3])"
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

    with (open(file_path, 'r+') as f):
        matrix_input = Matrix(1, input_layer)
        matrix_goal = Matrix(1, output_layer)

        lines = f.readlines()
        assert (input_layer == int(lines[0])), "Value error! with input_layer == int(lines[0])"
        input_vector = lines[2]
        goal_vector = lines[4]

        goal_numbers = input_vector.strip().split()
        goal_numbers = [float(num) for num in goal_numbers]
        for num in range(input_layer):
            matrix_goal.set(0, num, goal_numbers[num])

        input_numbers = input_vector.strip().split()
        input_numbers = [float(num) for num in input_numbers]
        for num in range(input_layer):
            matrix_input.set(0, num, input_numbers[num])

        # 计算所得矩阵
        h = matrix_input.mul(hidden_first_weight)
        for k in range(num_of_plies-1):
            h = h.mul(hidden_weight[k])
        matrix_output = h.mul(hidden_last_weight)

        # 写入矩阵
        f.write("\n\n\n"+"以下为结果向量\n\n")
        for i in range(output_layer):
            f.write(str(matrix_output.matrix[0][i])+"\t\t\t")
        f.write(f"\n与预期矩阵方差为{matrix_output.variance(matrix_goal)}\n")
        print("写入所得矩阵完毕")
