import random
class sym_regression:
    def __init__(self, x, f_x):
        self.x = x
        self.f_x = f_x
        self.n = len(x)

    def cross_over(self,other):
        new_x = []
        new_f_x = []
        for i in range(self.n):
            if i % 2 == 0:
                new_x.append(self.x[i])
                new_f_x.append(self.f_x[i])
            else:
                new_x.append(other.x[i])
                new_f_x.append(other.f_x[i])
        return sym_regression(new_x, new_f_x)
    def mutate(self, mutation_rate):
        for i in range(self.n):
            if random.random() < mutation_rate:
                self.f_x[i] += random.uniform(-1, 1)