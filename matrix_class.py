import random
import math


class Matrix:

    def __init__(self, line, column, type_of_member=float):
        self.type_of_member = type_of_member
        self.line = line
        self.column = column
        self.matrix = [[0.0 for _ in range(column)] for _ in range(line)]
        for j in range(line):
            for i in range(column):
                self.matrix[j][i] = round(random.uniform(-1, 1), 10)

    def set(self, line, column, value):
        value_ = float(value)
        assert (isinstance(value_, self.type_of_member)), f"Type error! with {value} is not {self.type_of_member}"
        if value_ > 1:
            value_ = 1
        if value_ < -1:
            value_ = -1
        assert (-1 <= value_ <= 1), f"Value error! value={value_}>1 or <-1"
        self.matrix[line][column] = value_

    def mul(self, matrix2):
        assert (self.column == matrix2.line), "Value error:self.column == matrix2.line"
        x = Matrix(self.line, matrix2.column)    # 所得矩阵
        for i in range(self.line):
            for j in range(matrix2.column):
                summ = 0.0
                for k in range(self.column):
                    assert (i < self.line and k < self.column and j < matrix2.column)
                    summ += (self.matrix[i][k])*(matrix2.matrix[k][j])
                summ /= self.column
                if summ >= 0:
                    summ = math.sqrt(summ)
                else:
                    summ = -1*math.sqrt(-summ)
                x.set(i, j, round(summ, 10))
        return x

    def sub(self, matrix2):
        assert (self.column == matrix2.column), "Value error:self.column == matrix2.column"
        assert (self.line == matrix2.line), "Value error:self.line == matrix2.line"
        x = Matrix(self.line, matrix2.column)
        for i in range(matrix2.line):
            for j in range(matrix2.column):
                x.set(i, j, round((self.matrix[i][j]-matrix2.matrix[i][j]), 10))
        return x

    def plus(self, matrix2):
        assert (self.column == matrix2.column), "Value error:self.column == matrix2.column"
        assert (self.line == matrix2.line), "Value error:self.line == matrix2.line"
        x = Matrix(self.line, matrix2.column)
        for i in range(matrix2.line):
            for j in range(matrix2.column):
                x.set(i, j, round((self.matrix[i][j]+matrix2.matrix[i][j]), 10))
        return x

    def mul_num(self, num):
        for i in range(self.line):
            for j in range(self.column):
                self.set(i, j, round(self.matrix[i][j]*num, 10))
        return self

    def transpose(self):
        x = Matrix(self.column, self.line)
        for i in range(self.line):
            for j in range(self.column):
                x.matrix[j][i] = round(self.matrix[i][j], 10)
        return x

    def variance(self, matrix2):
        assert (self.column == matrix2.column), "Value error:self.column == matrix2.column"
        assert (self.line == matrix2.line), "Value error:self.line == matrix2.line"
        summ = 0
        x = self.sub(matrix2)
        for i in range(self.column):
            for j in range(self.line):
                summ += (x.matrix[j][i])*(x.matrix[j][i])
        summ /= (self.column * self.line)
        return summ
