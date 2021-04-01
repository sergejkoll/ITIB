from typing import List
import numpy as np
import matplotlib.pyplot as plt


class Neuron(object):
    def __init__(self, a: float, b: float, size_p: int, N: int, learning_rate: float, number_of_epochs: int):
        if size_p <= 0 or size_p > N:
            raise ValueError
        if learning_rate > 1 or learning_rate <= 0:
            raise ValueError
        if a > b:
            raise ValueError

        self.a = a
        self.b = b
        self.p = size_p
        self.learning_rate = learning_rate
        self.N = N
        self.number_of_epochs = number_of_epochs

        self.weights = [0 for _ in range(size_p + 1)]
        self.real_values = None
        self.learning_values = None

    def get_X(self, a=None, b=None, N=None) -> List:
        points = []
        if a is None and b is None:
            step_in_interval = (self.b - self.a) / self.N
            for i in range(self.N):
                points.append(self.a + i * step_in_interval)
        else:
            step_in_interval = (b - a) / N
            for i in range(N):
                points.append(a + i * step_in_interval)

        return points

    def get_Y(self, a=None, b=None, N=None) -> List:
        points_Y = []
        if a is None and b is None:
            points_X = self.get_X()
        else:
            points_X = self.get_X(a, b, N)

        for x in points_X:
            points_Y.append(np.sin(2 * np.sqrt(np.exp(x))))

        return points_Y

    def get_delta(self, idx: int):
        return self.real_values[idx] - self.learning_values[idx]

    def get_epsilon(self):
        epsilon = 0
        for i in range(self.p, self.N):
            delta = self.get_delta(i)
            epsilon += delta ** 2
        return np.sqrt(epsilon)

    def learning(self):
        self.real_values = self.get_Y()
        self.learning_values = [0 for _ in range(len(self.real_values))]
        epsilon = 0
        for epoch in range(self.number_of_epochs):
            for point in range(self.p, self.N):
                net = self.weights[self.p]
                for i in range(self.p):
                    net += self.weights[i] * self.real_values[point - self.p + i]
                self.learning_values[point] = net
                for k in range(self.p):
                    self.weights[k] += self.learning_rate * self.get_delta(point) * self.real_values[point - self.p + k]

            epsilon = self.get_epsilon()
            if epsilon < 0.001:
                print(epsilon)
                break

        print(f"Epoch: {self.number_of_epochs}, eps: {epsilon}, weights: {self.weights}")

    def predict(self):
        for i in range(self.N, 2 * self.N):
            net = self.weights[self.p]
            for j in range(self.p):
                net += self.weights[j] * self.learning_values[i - self.p + j]
            self.learning_values.append(net)

    def plot_after_learning(self):
        plt.title("График после обучения")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        X = self.get_X()
        plt.plot(X, self.real_values, X[self.p:], self.learning_values[self.p:], 'ro')
        plt.show()

    def plot_predict(self):
        plt.title("График на следующем интервале")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        X = self.get_X(self.a, self.b * 2 - self.a, self.N * 2)
        Y = self.get_Y(self.a, self.b * 2 - self.a, self.N * 2)
        plt.plot(X, Y, X[self.p:], self.learning_values[self.p:], 'ro')
        plt.show()
