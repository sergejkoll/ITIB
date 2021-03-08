from abc import ABC, abstractmethod
from typing import List
import itertools
import math
import matplotlib.pyplot as plt


class Neuron(ABC):
    """
    Абстрактный класс нейрона

    Аргументы:
    weights -- массив начальных весов
    learning_rate -- норма обучения (0; 1]
    t -- целевой выход

    Атрибуты:
    X -- массив всех наборов от 4 переменных
    """

    def __init__(self, weights: List, learning_rate: float, t: List):
        if len(weights) != 5:
            raise ValueError
        self.weights = weights

        if learning_rate > 1 or learning_rate <= 0:
            raise ValueError
        self.learning_rate = learning_rate

        if len(t) != 16:
            raise ValueError
        self.t = t

        self.X = list(itertools.product([0, 1], repeat=4))

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def feedforward(self, x) -> int:
        pass

    def result(self) -> List[int]:
        """
        Функция которая считает результат на каждом наборе(X[i]) в зависимости от функции активации
        и весов в данной эпохе
        :return: результат функции текущей эпохи
        """
        func = []
        for i in range(16):
            func.append(self.feedforward(self.X[i]))
        return func

    def hamming_distance(self, y: List[int]) -> int:
        """
        Функция считает расстояние Хемминга
        :param y: вектор функции с которой сравнивается целевой вектор t
        :return: расстояние Хемминга
        """
        dist = 0
        for i in range(16):
            dist += y[i] ^ self.t[i]
        return dist

    @staticmethod
    def plot(epochs: int, E: List, title: str):
        plt.title(title)
        plt.xlabel("k")
        plt.ylabel("E(k)")
        plt.grid()
        plt.plot(range(epochs + 1), E)
        plt.show()


class NeuronWithUnitStepFunction(Neuron):
    """
    Класс реализующий нейрон с пороговой функцией активации
    """

    def train(self):
        """
        Функция обучения
        """
        epochs = 100
        E_list = []
        for epoch in range(epochs):
            Y = self.result()
            E = self.hamming_distance(Y)
            print(f"Epoch {epoch}: Y = {Y}, W = [{self.weights[0]:.3f}, {self.weights[1]:.3f}, {self.weights[2]:.3f}, "
                  f"{self.weights[3]:.3f}, {self.weights[4]:.3f}] E = {E}")
            E_list.append(E)
            if E == 0:
                self.plot(epoch, E_list, "Зависимоть E от k, задание №1")
                break
            for i in range(5):
                for j in range(16):
                    if i == 0:
                        self.weights[i] += self.learning_rate * (self.t[j] - Y[j]) * 1
                    else:
                        self.weights[i] += self.learning_rate * (self.t[j] - Y[j]) * self.X[j][i - 1]

    def feedforward(self, x) -> int:
        """
        Функция вычисления на наборе x значения функии f(net)
        :param x: набор на котором необходимо посчитать функцию
        :return: значение f(net)
        """
        net = 0
        for i, w in enumerate(self.weights[1:]):
            net += w * x[i]
        net += self.weights[0]
        return 1 if net >= 0 else 0


class NeuroneWithSigmoidActivationFunction(Neuron):
    """
    Класс реализующий нейрон с сигмоидальной функцией активации
    """

    def train(self, X=None, logging=True) -> bool:
        """
        Функция обучения нейрона
        :param X: наборы обучающей выборки
        :param logging: печать в консоль информацию об эпохе (Да/Нет)
        :return: True - обучение завершилось успехов за данное число эпох, иначе False
        """
        if X is None:
            X = self.X

        epochs = 100
        E_list = []
        for epoch in range(epochs):
            Y = self.result()
            E = self.hamming_distance(Y)
            if logging:
                print(f"Epoch {epoch}: Y = {Y}, W = [{self.weights[0]:.3f}, {self.weights[1]:.3f},"
                      f"{self.weights[2]:.3f}, {self.weights[3]:.3f}, {self.weights[4]:.3f}] E = {E}")
            E_list.append(E)
            if E == 0:
                if logging:
                    self.plot(epoch, E_list, "Зависимость E от k, задания №2 и №3")
                return True
            for i in range(5):
                for j in range(len(X)):
                    if i == 0:
                        self.weights[i] += self.learning_rate * (self.t[j] - Y[j]) * self.sigmoid_function(X[j]) \
                                           * (1 - self.sigmoid_function(X[j])) * 1
                    else:
                        self.weights[i] += self.learning_rate * (self.t[j] - Y[j]) * self.sigmoid_function(X[j]) \
                                           * (1 - self.sigmoid_function(X[j])) * X[j][i - 1]

        return False

    def train_partly(self):
        """
        Обучение на неполной выборке (проходим по всем возможным сочетаниям из X начиная с 2 элементов)
        и останавливаемся на первой минимальной
        """
        for i in range(2, 16):
            combinations = itertools.combinations(self.X, i)
            flag = False
            for item in combinations:
                self.weights = [0, 0, 0, 0, 0]
                res = self.train(item, False)
                if res:
                    flag = True
                    print(f"Набор из {i} векторов: {item}")
                    self.weights = [0, 0, 0, 0, 0]  # Повтор с логированием
                    self.train(item, True)  # Повтор с логированием
                    break
            if flag:
                break

    def feedforward(self, x) -> int:
        """
        Вычисление y(out)
        :param x: набор на котором необходимо вычислить функцию
        :return: y(out)
        """
        return 1 if self.sigmoid_function(x) >= 0.5 else 0

    def sigmoid_function(self, x):
        """
        Вычисление out (сигмоидальной функции)
        :param x: набор на котором необходимо посчитать функцию
        :return: значение сигмоидальной функции f(net) на наборе x
        """
        net = 0
        for i, w in enumerate(self.weights[1:]):
            net += w * x[i]
        net += self.weights[0]
        return 1 / (1 + math.exp(-net))
