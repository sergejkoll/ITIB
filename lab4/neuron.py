from typing import List
import numpy as np
import matplotlib.pyplot as plt
import itertools


class Neuron(object):
    def __init__(self, target: List, learning_rate: float, sigmoid_function: bool):
        if learning_rate <= 0 or learning_rate > 1:
            raise ValueError
        if np.log2(len(target)) != 4:
            raise ValueError

        self.target = target
        self.learning_rate = learning_rate
        self.sigmoid_function = sigmoid_function

        self.synaptic_weights, self.set_indices = self.get_synaptic_weights()
        self.center_coordinates = self.get_center_coordinates()
        self.sample = list(itertools.product([0, 1], repeat=4))

    def get_synaptic_weights(self):
        idxOnes = []
        count_ones = 0
        idxZeros = []
        count_zeros = 0
        for idx, element in enumerate(self.target):
            if element == 0:
                count_zeros += 1
                idxZeros.append(idx)
            else:
                count_ones += 1
                idxOnes.append(idx)

        if count_ones < count_zeros:
            return [0 for _ in range(count_ones + 1)], idxOnes
        elif count_zeros < count_ones:
            return [0 for _ in range(count_zeros + 1)], idxZeros

    def get_center_coordinates(self) -> List:
        center_coordinates = []
        for idx in self.set_indices:
            vec = [int(x) for x in bin(idx)[2:]]
            if len(vec) < 4:
                while len(vec) != 4:
                    vec.insert(0, 0)
            center_coordinates.append(vec)
        return center_coordinates

    @staticmethod
    def GaussianRBF(X: List, C: List) -> float:
        sum = 0
        for i in range(4):
            sum += (X[i] - C[i]) ** 2
        return np.exp(-sum)

    @staticmethod
    def bin_to_dec(number):
        dec = 0
        for i in range(len(number)):
            dec += int(number[i]) * (2 ** (len(number) - i - 1))

        return dec

    def net(self, X: List) -> float:
        result = self.synaptic_weights[len(self.center_coordinates)]
        for j, cj in enumerate(self.center_coordinates):
            result += self.synaptic_weights[j] * self.GaussianRBF(X, cj)
        return result

    def getY(self, X: List) -> int:
        if self.sigmoid_function:
            return 1 if (1 / (1 + np.exp(-self.net(X)))) >= 0.5 else 0
        else:
            return 1 if self.net(X) >= 0 else 0

    def real_out(self) -> List:
        func = []
        for i in range(16):
            X = [int(x) for x in bin(i)[2:]]
            if len(X) < 4:
                while len(X) != 4:
                    X.insert(0, 0)
            func.append(self.getY(X))
        return func

    def hamming_distance(self, y: List[int]) -> int:
        dist = 0
        for i in range(16):
            dist += y[i] ^ self.target[i]
        return dist

    def learning(self, training_sample, with_log: bool) -> bool:
        epochs = 300
        E_list = []
        for epoch in range(epochs):
            Y = self.real_out()
            E = self.hamming_distance(Y)
            E_list.append(E)
            if with_log:
                print(f"Epoch {epoch}: Y = {Y}, W = {self.synaptic_weights}, C = {self.center_coordinates}, E = {E}")
            if E == 0:
                if with_log:
                    self.plot(epoch, E_list)
                return True

            for i in range(len(training_sample)):
                y = self.getY(training_sample[i])
                delta = self.target[self.bin_to_dec(training_sample[i])] - y
                for j, cj in enumerate(self.center_coordinates):
                    RBF_out = self.GaussianRBF(training_sample[i], cj)
                    self.synaptic_weights[j] += self.learning_rate * delta * RBF_out
                self.synaptic_weights[len(self.center_coordinates)] += self.learning_rate * delta

        return False

    def learning_partly(self):
        for i in range(2, 16):
            combinations = itertools.combinations(self.sample, i)
            flag = False
            for item in combinations:
                self.synaptic_weights = [0 for _ in range(len(self.set_indices) + 1)]
                successful_learning = self.learning(item, False)
                if successful_learning:
                    flag = True
                    print(f"Набор из {i} векторов: {item}")
                    self.synaptic_weights = [0 for _ in range(len(self.set_indices) + 1)]  # Повтор с логированием
                    self.learning(item, True)  # Повтор с логированием
                    break
            if flag:
                break

    @staticmethod
    def plot(epochs: int, E: List):
        plt.xlabel("k")
        plt.ylabel("E(k)")
        plt.grid()
        plt.plot(range(epochs + 1), E)
        plt.show()
