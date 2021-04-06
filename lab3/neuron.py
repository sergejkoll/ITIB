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

        self.x_train = [a + i * (b - a) / N for i in range(N + 1)]
        self.x_predict = [b + i * (b - a) / N for i in range(1, N)]
        self.y_train = [self.f(x) for x in self.x_train]
        self.y_predict = [self.f(x) for x in self.x_predict]

        self.weights = [0 for _ in range(size_p + 1)]

    @staticmethod
    def f(x):
        return np.sin(2 * np.sqrt(np.exp(x)))

    def get_X_predict(self) -> List:
        return self.x_train[-1:] + self.x_predict

    def get_Y_predict(self) -> List:
        return self.y_train[-self.p - 1:] + self.y_predict

    def get_windows(self, x: List):
        return [[x[i] for i in range(bias, bias + self.p)] for bias in range(len(x) - self.p)]

    def get_predict_windows(self):
        return [[self.y_train[i] for i in range(bias, bias + self.p)]
                for bias in range(len(self.y_train) - self.p)]

    def learning(self):
        windows = self.get_windows(self.y_train)
        targets = self.y_train[self.p:]
        epsilon = 0
        learning_values = []
        for epoch in range(self.number_of_epochs):
            learning_values = []
            total_error = 0
            for (window, target) in zip(windows, targets):
                x = [1] + window
                out = sum([self.weights[i]*x[i] for i in range(len(self.weights))])
                learning_values.append(out)
                error = target - out
                for i in range(0, len(self.weights)):
                    self.weights[i] += self.learning_rate * error * x[i]
                total_error += pow(error, 2)
            epsilon = np.sqrt(total_error)

        print(epsilon)
        print(self.weights)

        plt.title("График после обучения")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        x_train = self.x_train[self.p:]
        plt.plot(self.x_train, self.y_train, x_train, learning_values, 'ro')
        plt.show()

    def predict(self):
        x_ = self.get_X_predict()
        y = self.get_Y_predict()
        windows = self.get_windows(y)
        predictions = []
        for window in windows:
            x = [1] + window
            prediction = sum([self.weights[i]*x[i] for i in range(len(self.weights))])
            predictions.append(prediction)

        y = y[self.p:]
        plt.title("График на следующем интервале")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.plot(x_, y, x_, predictions, 'ro')
        plt.show()

    # def plot_after_learning(self):
    #     plt.title("График после обучения")
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.grid()
    #     X = self.get_X()
    #     plt.plot(X, self.real_values, X[self.p:], self.learning_values[self.p:], 'ro')
    #     plt.show()

    # def plot_predict(self):
    #     plt.title("График на следующем интервале")
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.grid()
    #     X = self.get_X(self.a, self.b * 2 - self.a, self.N * 2)
    #     Y = self.get_Y(self.a, self.b * 2 - self.a, self.N * 2)
    #     plt.plot(X, Y, X[self.p:], self.learning_values[self.p:], 'ro')
    #     plt.show()
