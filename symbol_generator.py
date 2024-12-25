import random
#  说明：生成的文档第一行为行数，第二行为列数，第三行为矩阵个数。数据存储方式为先存储原矩阵再存储乘二矩阵

# 如果要改变训练目标规则，修改样本生成、测试生成方法

def generate_vector_goal(file_path, input_layer, output_layer, num_of_sample):
    with open(file_path, 'w') as f:
        f.writelines(str(input_layer)+"\n")
        f.writelines(str(output_layer)+"\n")
        f.writelines(str(num_of_sample)+"\n")
        for i in range(num_of_sample):
            x = [round(random.uniform(-1, 1), 10) for _ in range(input_layer)]
            y = [0.0 for _ in range(output_layer)]
            for j in range(input_layer):
                y[j] = round(x[j]/2+random.uniform(-0.0000001, 0.0000001), 10)
            for j in range(input_layer):
                f.write(str(x[j])+"\n")
            for j in range(output_layer):
                f.write(str(y[j])+"\n")


def generate_matrix_goal(file_path, line, column, num_of_sample):
    with open(file_path, 'w') as f:
        f.writelines(str(line)+"\n")
        f.writelines(str(column)+"\n")
        f.writelines(str(num_of_sample)+"\n")
        for i in range(num_of_sample):
            x = [round(random.uniform(-1, 1), 10) for _ in range(column * line)]
            for j in range(column*line):
                f.write(str(x[j])+"\n")
            for j in range(column*line):
                f.write(str(round(x[j]/2+random.uniform(-0.0000001, 0.0000001), 10))+"\n")


def generate_vector_test(file_path, input_layer):
    x = [round(random.uniform(-1, 1), 10) for _ in range(input_layer)]
    with open(file_path, 'w') as f:
        f.writelines(str(input_layer)+"\n")
        f.writelines("this is input vector\n")
        for i in range(input_layer):
            f.write(str(x[i])+"\t")
        f.writelines("\nthis is goal vector\n")
        for i in range(input_layer):
            f.write(str(round(x[i]/2, 10))+"\t")
        f.write("\n")


def generate_matrix_test(file_path, line, column):
    x = [round(random.uniform(-1, 1), 10) for _ in range(column * line)]
    with open(file_path, 'w') as f:
        f.writelines(str(line)+"\n")
        f.writelines(str(column)+"\n")
        f.writelines("this is input matrix\n")
        for j in range(column*line):
            f.write(str(x[j])+"\t")
        f.writelines("this is goal matrix\n")
        for j in range(column*line):
            f.write(str(round(x[j]/2, 10))+"\t")
        f.write("\n")


def generate_vector_weight(file_path, input_layer, hidden_layer, output_layer, num_of_plies):
    with open(file_path, 'w') as f:
        f.writelines(str(input_layer)+"\n")
        f.writelines(str(hidden_layer)+"\n")
        f.writelines(str(output_layer)+"\n")
        f.writelines(str(num_of_plies)+"\n")
        for j in range(input_layer):
            for k in range(hidden_layer):
                f.write(str(round(random.uniform(-1, 1), 10))+"\n")
        for i in range(num_of_plies-1):
            for j in range(hidden_layer):
                for k in range(hidden_layer):
                    f.write(str(round(random.uniform(-1, 1), 10))+"\n")
        for j in range(hidden_layer):
            for k in range(output_layer):
                f.write(str(round(random.uniform(-1, 1), 10))+"\n")


def generate_matrix_weight(file_path, num_of_hidden_plies, line_1, column_1, column_2):
    with open(file_path, 'w') as f:
        for i in range(num_of_hidden_plies):
            for j in range(line_1):
                for k in range(column_1):
                    f.write(str(round(random.uniform(-1, 1), 10))+"\n")
            for j in range(column_1):
                for k in range(column_2):
                    f.write(str(round(random.uniform(-1, 1), 10))+"\n")


if __name__ == '__main__':
    # generate_matrix_goal("data/goal_matrix.txt")
    # generate_matrix_weight("data/Weight_matrix.txt",500, 20, 20, 30)
    # generate_matrix_test("data/test_matrix.txt", 20, 30)
    generate_vector_goal("data/goal_vector.txt", 20, 20, 1)
    generate_vector_weight("data/Weight_vector.txt", 20, 20, 20, 1)
    generate_vector_test("data/test_vector.txt", 20)
